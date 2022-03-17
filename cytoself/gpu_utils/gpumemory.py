def gpuinfo(gpuidx):
    """
    Get GPU information
    :param gpuidx: GPU index
    :return: GPU information in dictionary
    """
    import subprocess

    sp = subprocess.Popen(
        ['nvidia-smi', '-q', '-i', str(gpuidx), '-d', 'MEMORY'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('BAR1', 1)[0].split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass
    return out_dict


def getfreegpumem(gpuidx):
    """
    Get free GPU memory
    :param gpuidx: GPU ID
    :return: Free memory size
    """
    return int(gpuinfo(gpuidx)['Free'].replace('MiB', '').strip())


def getbestgpu(log_level=0):
    """
    Get the GPU ID with the largest free memory
    :param log_level: level to print message; 0: print everything, 1: only print selected device, 2: print nothing
    :return: the GPU ID with the largest free memory
    """
    freememlist = []
    for gpuidx in range(4):
        freemem = getfreegpumem(gpuidx)
        freememlist.append(freemem)
        if log_level == 0:
            print("GPU device %d has %d MiB left." % (gpuidx, freemem))
    idbest = freememlist.index(max(freememlist))
    if log_level < 2:
        print("--> GPU device %d was chosen" % idbest)
    return idbest
