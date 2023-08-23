import logging, sys, os

def use_file(path):

    logger = logging.getLogger()
    logging.basicConfig(stream=(sys.stderr), level=(logging.DEBUG))
    
    # Disable matplotlib warning messages
    logging.getLogger('matplotlib.font_manager').disabled = True

    # For not showing PIL DEBUG message
    logging.getLogger('PIL').setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    filepath = os.path.join(path, 'output.log')
    file_handler = logging.FileHandler(filepath, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # logging.info('Logging to file mode started. All logging events will be save to the file %s', 'syslog.log')
    logger.addHandler(file_handler)
    return file_handler    


