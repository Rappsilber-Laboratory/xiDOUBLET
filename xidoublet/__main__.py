from multiprocessing import Process, Queue, Value, cpu_count
import queue as Q
from optparse import OptionParser
from .detector import SearchContext, SpectrumProcessor
from .result_writer import TsvResultWriter
from .xi_logging import log_enable, progress_enable, log, ProgressBar
from .config import Config, ConfigReader
from .spectra_reader import PeakListWrapper
from multiprocessing import Pool
import sys
import os
from pathlib import Path
import traceback


parser = OptionParser()
parser.add_option("-i", "--input", dest="peaklist", action="append",
                  help="Peaklist file with spectra to search",
                  metavar="PEAK")
parser.add_option("-o", "--output", dest="out_dir",
                  help="Directory to output results",
                  metavar="OUT")
parser.add_option("-n", "--basename", dest="basename",
                  help="output file base name",
                  metavar="BASE", default='')
parser.add_option("-c", "--config", dest="config",
                  help="JSON configuration file",
                  metavar="JSON")
parser.add_option("-b", "--no-bar", dest="progress_bar",
                  default=True, action='store_false',
                  help="Show progress bar")
parser.add_option("-O", "--config-option", dest="additional_config_options",
                  default=[], action='append',
                  help="additional config-option to be evaluated")

(options, args) = parser.parse_args()

log_enable(True)
progress_enable(options.progress_bar)


def fail(message):
    print(message)
    sys.exit(1)


def validate_options(options):
    if not options.peaklist:
        fail("peaklist file_name is required")
    if not options.out_dir:
        fail("output directory is required")


def write_queue(queue, keep_writing, result_writer):
    while keep_writing.value:
        try:
            s = queue.get(True, 1)
            while not s.size == 0:
                result_writer.write(s)
                s = queue.get(False)
        except Q.Empty:
            ...
    result_writer.close()


if options.config:
    config = ConfigReader.load_file(options.config)
else:
    config = Config()

for o in options.additional_config_options:
    exec("config." + o)

context = SearchContext(config)
processor = SpectrumProcessor(context)


# define function to process each spectra
if options.progress_bar:
    def main(spectrum):
        try:
            processor.process_spectrum(spectrum, doublet_queue)
            bar_queue.put(True)
        except Exception:
            traceback.print_exc()
            return False
        return True

else:
    def main(spectrum):
        try:
            processor.process_spectrum(spectrum, doublet_queue)
        except Exception:
            traceback.print_exc()
            return False
        return True


def setup_queued_writing(out_path):
    writer = TsvResultWriter(out_path)
    # The multiprocessing queue that we use to get data to the writer. The Spectrum
    # Processors will be writing to this and the queue_writer reads from it.
    queue = Queue()
    # This is a shared value to be used with the writer process: it allows us to let
    # the writer know that we are done processing and the writer can close itself.
    keep_writing = Value('b', True)
    # Setting 'daemon=True' below means that if the main process dies (eg, because of
    # an error) the writer will die as well. This means, however, that we must take
    # care to deal with the writer carefully or we could lose data. That is why we
    # call writer_process.join() at the end of this method.
    writer_process = Process(target=write_queue, args=(queue, keep_writing, writer), daemon=True)
    writer_process.start()
    return queue, keep_writing, writer_process


if __name__ == '__main__':
    # Validate argument combinations
    validate_options(options)

    # make sure the output directory exists
    Path.mkdir(Path(options.out_dir), exist_ok=True, parents=True)

    doublet_out = os.path.join(options.out_dir, f'{options.basename}doublets.tsv')
    doublet_queue, doublet_keep_writing, doublet_writer_process = setup_queued_writing(doublet_out)

    log("Loading spectra")
    spectra_wrapper = PeakListWrapper(context)
    spectra_wrapper.load(options.peaklist)
    num_spectra = spectra_wrapper.count_spectra()
    log("Loaded %d spectra, starting search" % num_spectra)

    if options.progress_bar:
        bar_queue = Queue()
        bar = ProgressBar("Processing spectra", num_spectra)

        def bar_updater(queue):
            while True:
                s = queue.get()
                if s:
                    bar.next()
                else:
                    return
        bar_process = Process(target=bar_updater, args=(bar_queue,), daemon=True)
        bar_process.start()

    thread_count = config.threads
    if thread_count == 0:
        thread_count = None  # multithreading doesn't like 0 as a value
    elif thread_count < 0:
        thread_count = cpu_count() + thread_count
    if thread_count == 1:
        for s in spectra_wrapper.spectra:
            main(s)
    else:
        thread_pool = Pool(thread_count)
        # process spectra across threads
        for r in thread_pool.imap_unordered(main, spectra_wrapper.spectra, 10):
            if not r:
                raise Exception(
                    'Exception occurred during main search! See stack trace above for details.')

        # https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html#if-you-use-multiprocessing-pool
        thread_pool.close()  # Marks the pool as closed.
        thread_pool.join()  # Waits for workers to exit.

    # end progress bar
    if options.progress_bar:
        bar.finish()

    # tell the writers we're done and wait for it to stop
    doublet_keep_writing.value = False

    # Not sure why we need the next 5 lines, just a `writer_process.join()` should be
    # enough. But for some reason, the writer_process doesn't always die, and then the
    # `join` doesn't return.
    log("Search finished, waiting for result writer to complete...")
    doublet_queue.close()
    doublet_queue.join_thread()
    doublet_writer_process.join(1)
    while doublet_queue.qsize() > 0:
        doublet_writer_process.join(1)

    log("Finished search!")
