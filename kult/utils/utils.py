import os
import errno

def is_file_ready(file_path):
    try:
        # Попытка открыть файл в режиме "эксклюзивного" доступа
        with open(file_path, 'ab') as f:
            os.lockf(f.fileno(), os.F_TEST, 0)
        return True
    except IOError as e:
        if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
            return False
        raise