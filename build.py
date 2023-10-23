import fnmatch
import platform
import hashlib
import os
import sys
import shutil
import urllib.parse as urlparse
import yaml
import requests

from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory as tempdir

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    # Previously we used to send a HEAD request to the URL and get the size
    # of the file before we downloaded it - this helped with creating a nice
    # progress bar using tqdm. But now this is broken - nvidia's servers don't
    # respect HEAD requests return incorrect content-length; this results
    # in incomplete files. Removing tqdm-based progress bars for now.
    print("Downloading %s to %s" % (url, dst))
    req = requests.get(url, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    f.close()
    s = os.path.getsize(dst)
    print("File downloaded: {0}".format(dst))
    print("File size (bytes): {0}".format(s))
    return s

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# The config dictionary looks like:
# config[cuda_version(s)...]
#
# and for each cuda_version the keys:
# base_url the base url for all downloads
# patch_url_ext the extra path needed to reach the patch directory from base_url
# installers_url_ext the extra path needed to reach the local installers directory
# md5_url the url for checksums
# cuda_libraries the shared libraries to copy in
# cuda_static_libraries the static libraries to copy in
# libdevice_versions the library device versions supported (.bc files)
# linux the linux platform config (see below)
# windows the windows platform config (see below)
#
# For each of the two platform specific dictionaries, linux and windows
# a dictionary containing keys:
# blob the name of the downloaded file, for linux this is the .run file
# patches a list of the patch files for the blob, they are applied in order
# cuda_lib_fmt string format for the cuda libraries
# nvvm_lib_fmt string format for the nvvm libraries
# libdevice_lib_fmt string format for the libdevice.compute bitcode file
#
# To accommodate nvtoolsext not being present as a DLL in the installer PE32s on windows,
# the windows variant of this script supports assembly directly from a pre-installed
# CUDA toolkit. The environment variable "NVTOOLSEXT_INSTALL_PATH" can be set to the
# installation path of the CUDA toolkit's NvToolsExt location (this is not the user
# defined install directory) and the DLL will be taken from that location.

###########################################
####### CUDA 12.3.0 setup (Oct 2023) ######
###########################################

# Create the config object.
config = {}
# Package version declaration must match CUDA release version.
# From release 11 onwards, the format major.minor.micro is always followed.
config['version'] = os.environ['PKG_VERSION']
config['base_url'] = "https://developer.download.nvidia.com/compute/cuda/{0}/".format(config['version'])
config['installers_url_ext'] = 'local_installers/'
config['patch_url_ext'] = ''
config['md5_url'] = "{0}docs/sidebar/md5sum.txt".format(config['base_url'])
config['cuda_libraries'] = [
    'cublas',
    'cublasLt',
    'cudart',
    'cufft',
    'cufftw',
    'curand',
    'cusolver',
    'cusolverMg',
    'cusparse',
    'nppc',
    'nppial',
    'nppicc',
    'nppidei',
    'nppif',
    'nppig',
    'nppim',
    'nppist',
    'nppisu',
    'nppitc',
    'npps',
    'nvToolsExt',
    'nvblas',
    'nvjpeg',
    'nvrtc',
    'nvrtc-builtins',
]
config['cuda_static_libraries'] = [
    'cudadevrt'
]
# accinj64 is only available on linux
if sys.platform.startswith('linux'):
    config['cuda_libraries'].append('accinj64')
    config['cuda_libraries'].append('cuinj64')
# cuinj is only available on windows
if sys.platform.startswith('windows'):
    config['cuda_libraries'].append('cuinj')
config['libdevice_versions'] = [config['version'].split(".")[0]]

config['linux'] = {
    'blob': 'cuda_12.3.0_545.23.06_linux.run',
    'ppc64le_blob': 'cuda_12.3.0_545.23.06_linux_ppc64le.run',
    'aarch64_blob': 'cuda_12.3.0_545.23.06_linux_sbsa.run',
    'patches': [],
    # need globs to handle symlinks
    'cuda_lib_fmt': 'lib{0}.so*',
    'cuda_static_lib_fmt': 'lib{0}.a',
    'nvtoolsext_fmt': 'lib{0}.so*',
    'nvvm_lib_fmt': 'lib{0}.so*',
    'libdevice_lib_fmt': 'libdevice.10.bc'
}

config['windows'] = {
    'blob': 'cuda_12.3.0_545.84_windows.exe',
    'patches': [],
    'cuda_lib_fmt': '{0}64_1*.dll',
    'cuda_static_lib_fmt': '{0}.lib',
    'nvtoolsext_fmt': '{0}64_1.dll',
    'nvvm_lib_fmt': '{0}64_40_0.dll',
    'libdevice_lib_fmt': 'libdevice.10.bc',
    'NvToolsExtPath' :
        os.path.join('c:' + os.sep, 'Program Files',
                    'NVIDIA Corporation', 'NVToolsExt', 'bin')
}

class Extractor(object):
    """Extractor base class, platform specific extractors should inherit
    from this class.
    """

    libdir = {'linux': 'lib',
              'windows': 'Library/bin'}

    def __init__(self, ver_config, plt_config):
        """Initialise an instance:
        Arguments:
          version - CUDA version string
          ver_config - the configuration for this CUDA version
          plt_config - the configuration for this platform
        """
        self.config_version = ver_config['version']
        self.md5_url = ver_config['md5_url']
        self.base_url = ver_config['base_url']
        self.patch_url_ext = ver_config['patch_url_ext']
        self.installers_url_ext = ver_config['installers_url_ext']
        self.cuda_libraries = ver_config['cuda_libraries']
        self.cuda_static_libraries = ver_config['cuda_static_libraries']
        self.libdevice_versions = ver_config['libdevice_versions']
        self.config_blob = plt_config['blob']
        self.embedded_blob = plt_config.get('embedded_blob', None)
        self.cuda_lib_fmt = plt_config['cuda_lib_fmt']
        self.cuda_static_lib_fmt = plt_config['cuda_static_lib_fmt']
        self.nvtoolsext_fmt = plt_config.get('nvtoolsext_fmt')
        self.nvvm_lib_fmt = plt_config['nvvm_lib_fmt']
        self.libdevice_lib_fmt = plt_config['libdevice_lib_fmt']
        self.patches = plt_config['patches']
        self.nvtoolsextpath = plt_config.get('NvToolsExtPath')
        self.config = {'version': self.config_version, **ver_config}
        self.prefix = os.environ['PREFIX']
        self.src_dir = os.environ['SRC_DIR']
        self.output_dir = os.path.join(self.prefix, self.libdir[getplatform()])
        self.symlinks = getplatform() == 'linux'
        self.debug_install_path = os.environ.get('DEBUG_INSTALLER_PATH')

        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            pass

    def download_blobs(self):
        """Downloads the binary blobs to the $SRC_DIR
        """
        dl_url = urlparse.urljoin(self.base_url, self.installers_url_ext)
        dl_url = urlparse.urljoin(dl_url, self.config_blob)
        dl_path = os.path.join(self.src_dir, self.config_blob)
        if not self.debug_install_path:
            download_from_url(dl_url, dl_path)
        else:
            existing_file = os.path.join(self.debug_install_path, self.config_blob)
            print("DEBUG: copying %s to %s" % (existing_file, dl_path))
            shutil.copy(existing_file, dl_path)

        for p in self.patches:
            dl_url = urlparse.urljoin(self.base_url, self.patch_url_ext)
            dl_url = urlparse.urljoin(dl_url, p)
            dl_path = os.path.join(self.src_dir, p)
            if not self.debug_install_path:
                download_from_url(dl_url, dl_path)
            else:
                existing_file = os.path.join(self.debug_install_path, p)
                print("DEBUG: copying %s to %s" % (existing_file, dl_path))
                shutil.copy(existing_file, dl_path)

    def check_md5(self):
        """Checks the md5sums of the downloaded binaries
        """
        md5file = self.md5_url.split('/')[-1]
        path = os.path.join(self.src_dir, md5file)
        download_from_url(self.md5_url, path)

        # Compute hash of blob.
        blob_path = os.path.join(self.src_dir, self.config_blob)
        md5sum = md5(blob_path)

        # Get checksums.
        with open(md5file, 'r') as f:
            checksums = [x.strip().split() for x in f.read().splitlines() if x]

        # Check md5 and filename match up.
        check_dict = {x[0]: x[1] for x in checksums}
        assert check_dict[md5sum].startswith(self.config_blob[:-7]), "md5sum doesn't match"

    def copy(self, *args):
        """The method to copy extracted files into the conda package platform
        specific directory. Platform specific extractors must implement.
        """
        raise RuntimeError('Must implement')

    def extract(self, *args):
        """The method to extract files from the cuda binary blobs.
        Platform specific extractors must implement.
        """
        raise RuntimeError('Must implement')

    def get_paths(self, libraries, dirpath, template):
        """Gets the paths to the various cuda libraries and bc files
        """
        pathlist = []
        for libname in libraries:
            filename = template.format(libname)
            paths = fnmatch.filter(os.listdir(dirpath), filename)
            if not paths:
                msg = ("Cannot find item: %s, looked for %s" %
                       (libname, filename))
                raise RuntimeError(msg)
            if (not self.symlinks) and (len(paths) != 1):
                msg = ("Aliasing present for item: %s, looked for %s" %
                       (libname, filename))
                msg += ". Found: \n"
                msg += ', \n'.join([str(x) for x in paths])
                raise RuntimeError(msg)
            pathsforlib = []
            for path in paths:
                tmppath = os.path.join(dirpath, path)
                assert os.path.isfile(tmppath), 'missing {0}'.format(tmppath)
                pathsforlib.append(tmppath)
            if self.symlinks: # deal with symlinked items
                # get all DSOs
                concrete_dsos = [x for x in pathsforlib
                                 if not os.path.islink(x)]
                # find the most recent library version by name
                target_library = max(concrete_dsos)
                # remove this from the list of concrete_dsos
                # all that remains are DSOs that are not wanted
                concrete_dsos.remove(target_library)
                # drop the unwanted DSOs from the paths
                [pathsforlib.remove(x) for x in concrete_dsos]
            pathlist.extend(pathsforlib)
        return pathlist

    def copy_files(self, cuda_lib_dir, nvvm_lib_dir, libdevice_lib_dir):
        """Copies the various cuda libraries and bc files to the output_dir
        """
        filepaths = []
        # nvToolsExt is different to the rest of the cuda libraries,
        # it follows a different naming convention, this accommodates...
        cudalibs = [x for x in self.cuda_libraries if x != 'nvToolsExt']
        filepaths += self.get_paths(cudalibs, cuda_lib_dir, self.cuda_lib_fmt)
        if 'nvToolsExt' in self.cuda_libraries:
            filepaths += self.get_paths(['nvToolsExt'], cuda_lib_dir,
                                        self.nvtoolsext_fmt)
        filepaths += self.get_paths(self.cuda_static_libraries, cuda_lib_dir,
                                    self.cuda_static_lib_fmt)
        filepaths += self.get_paths(['nvvm'], nvvm_lib_dir, self.nvvm_lib_fmt)
        filepaths += self.get_paths(self.libdevice_versions, libdevice_lib_dir,
                                    self.libdevice_lib_fmt)

        for fn in filepaths:
            if os.path.islink(fn):
                # replicate symlinks
                symlinktarget = os.readlink(fn)
                targetname = os.path.basename(fn)
                symlink = os.path.join(self.output_dir, targetname)
                print('linking %s to %s' % (symlinktarget, symlink))
                os.symlink(symlinktarget, symlink)
            else:
                print('copying %s to %s' % (fn, self.output_dir))
                shutil.copy(fn, self.output_dir)

    def dump_config(self):
        """Dumps the config dictionary into the output directory
        """
        dumpfile = os.path.join(self.output_dir, 'cudatoolkit_config.yaml')
        with open(dumpfile, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


class WindowsExtractor(Extractor):
    """The Windows extractor
    """

    def copy(self, *args):
        store, = args
        self.copy_files(
            cuda_lib_dir=store,
            nvvm_lib_dir=store,
            libdevice_lib_dir=store)

    def extract(self):
        runfile = self.config_blob
        patches = self.patches
        try:
            with tempdir() as tmpd:
                extract_name = '__extracted'
                extractdir = os.path.join(tmpd, extract_name)
                os.mkdir(extract_name)

                check_call(['7za', 'x', '-o%s' %
                            extractdir, os.path.join(self.src_dir, runfile)])
                for p in patches:
                    check_call(['7za', 'x', '-aoa', '-o%s' %
                                extractdir, os.path.join(self.src_dir, p)])

                nvt_path = os.environ.get('NVTOOLSEXT_INSTALL_PATH', self.nvtoolsextpath)
                print("NvToolsExt path: %s" % nvt_path)
                if nvt_path is not None:
                    if not Path(nvt_path).is_dir():
                        msg = ("NVTOOLSEXT_INSTALL_PATH is invalid "
                                "or inaccessible.")
                        raise ValueError(msg)

                # Put all the dlls into a directory named 'DLLs'.
                store_name = 'DLLs'
                store = os.path.join(tmpd, store_name)
                os.mkdir(store)
                for path, dirs, files in os.walk(extractdir):
                    if 'jre' not in path and 'GFExperience' not in path:  # don't get jre or GFExperience dlls
                        for filename in fnmatch.filter(files, "*.dll"):
                            if not Path(os.path.join(
                                    store, filename)).is_file():
                                shutil.copy(
                                    os.path.join(path, filename),
                                    store)
                        for filename in fnmatch.filter(files, "*.lib"):
                            if path.endswith('x64'):
                                if not Path(os.path.join(
                                        store, filename)).is_file():
                                    shutil.copy(
                                        os.path.join(path, filename),
                                        store)
                        for filename in fnmatch.filter(files, "*.bc"):
                            if not Path(os.path.join(
                                    store, filename)).is_file():
                                shutil.copy(
                                    os.path.join(path, filename),
                                    store)
                if nvt_path is not None:
                    for path, dirs, files in os.walk(nvt_path):
                        for filename in fnmatch.filter(files, "*.dll"):
                            if not Path(os.path.join(
                                    store, filename)).is_file():
                                shutil.copy(
                                    os.path.join(path, filename),
                                    store)
                self.copy(store)
        except PermissionError:
            # TODO: fix this
            # cuda 8 has files that refuse to delete, figure out perm changes
            # needed and apply them above, tempdir context exit fails to rmtree
            pass


class LinuxExtractor(Extractor):
    """The Linux extractor
    """

    def __init__(self, ver_config, plt_config):
        if platform.machine() == 'ppc64le':
            if plt_config.get('ppc64le_blob') is not None:
                plt_config['blob'] = plt_config['ppc64le_blob']
            else:
                raise RuntimeError('ppc64le not supported for %s' % ver_config['version'])

        super(LinuxExtractor, self).__init__(ver_config, plt_config)

    def copy(self, *args):
        basepath = args[0]
        self.copy_files(
            cuda_lib_dir=os.path.join(
                basepath, 'lib64'), nvvm_lib_dir=os.path.join(
                basepath, 'nvvm', 'lib64'), libdevice_lib_dir=os.path.join(
                basepath, 'nvvm', 'libdevice'))

    def extract(self):
        runfile = self.config_blob
        patches = self.patches
        os.chmod(runfile, 0o777)
        with tempdir(dir=os.getcwd()) as tmpd:
            if self.embedded_blob is not None:
                with tempdir() as tmpd2:
                    cmd = [os.path.join(self.src_dir, runfile),
                           '--extract=%s' % (tmpd2, ), '--nox11', '--silent']
                    check_call(cmd)
                    # extract the embedded blob
                    cmd = [os.path.join(tmpd2, self.embedded_blob),
                           '-prefix', tmpd, '-noprompt', '--nox11']
                    check_call(cmd)
            else:
                # Current Nvidia's Linux based runfiles don't use embedded runfiles
                #
                # "--installpath" runfile command is used to install the toolkit to a specified
                #     directory with the contents and layout similar to an install to
                #     '/usr/local/cuda`
                # "--override" runfile command to disable the compiler check since we are not
                #     installing the driver here
                # "--nox11" runfile command prevents desktop GUI on local install
                cmd = [os.path.join(self.src_dir, runfile),
                       '--installpath=%s' % (tmpd), '--toolkit', '--silent', '--override', '--nox11']
                check_call(cmd)
            for p in patches:
                os.chmod(p, 0o777)
                cmd = [os.path.join(self.src_dir, p),
                        '--installdir', tmpd, '--accept-eula', '--silent']
                check_call(cmd)
            self.copy(tmpd)

def getplatform():
    plt = sys.platform
    if plt.startswith('linux'):
        return 'linux'
    elif plt.startswith('win'):
        return 'windows'
    else:
        raise RuntimeError('Unknown platform')

dispatcher = {'linux': LinuxExtractor, 'windows': WindowsExtractor}

def _main():
    print("Running build - version {0}".format(config['version']))

    # Get an extractor
    plat = getplatform()
    extractor_impl = dispatcher[plat]
    extractor = extractor_impl(config, config[plat])

    # Download binaries
    print("Downloading binary blobs")
    extractor.download_blobs()

    # Check md5sum
    print("Comparing md5 hashes")
    extractor.check_md5()

    # Extract
    print("Extracting files")
    extractor.extract()

    # Dump config
    extractor.dump_config()

if __name__ == "__main__":
    _main()
