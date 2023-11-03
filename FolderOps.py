import tarfile
import os.path
import shutil

# 26 to 11 mbs
# so we can store 3 636 363.64 gage scope files on 20tb
def make_tarfile(target_directory, source_dir):
    with tarfile.open(target_directory + source_dir+'.tar.bz2', "w:bz2") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def kill_processed_file(source_dir):
    shutil.rmtree(source_dir)