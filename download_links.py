#!/bin/bash/python3

# A script to find all of the download links for cuda packages
# associated with NVIDIA Linux4Tegra
# This should be run on the Jetson device to get
# the appropriate packages for your device


import yaml
from subprocess import check_output


def recurse_links(library):

    output = {}


    stdout = check_output(["apt-get", 
                           "install", 
                           "--reinstall",
                           "--print-uris", 
                           "-qq",
                           library])
    if stdout:
        # this library has a deb file to be downloaded
        link, name, size, md5 = stdout.decode("utf-8").split()
        output[library] = {'link':link.replace('\'',''),
                           'name':name,
                           'size':int(size),
                           'md5':md5.split(':')[-1]}
    else:
        # library does not itself have a package to download, but probably has depends
        output[library] = {'link':None,'name':None,'md5':None}

    
    # Search for dependencies    
    stdout = check_output(["apt-cache","depends",library]).decode("utf-8").splitlines()
    Depends = [x.strip(" Depends:") for x in stdout if "Depends" in x]
    nvidia_depends = [x for x in Depends if any(word in x for word in ["cuda","nvidia","cublas"])]

    # Get the download links and md5s for those dependencies
    if nvidia_depends:
        [output.update(recurse_links(library)) for library in nvidia_depends]
    
    return output




# This function takes a long time to search for all of the links
with open(r'l4t_cuda.yaml','w') as file:
    file.write("# deb archives for CUDA 10.2 for Linux4Tegra \n")
    file.write("# information assembled using apt-cache depends <package>\n")
    file.write("# and apt-get install --reinstall --print-uris -qq <package> \n")
    yaml.dump(recurse_links("nvidia-cuda"),file)
