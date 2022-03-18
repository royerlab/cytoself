import torch


def gpuinfo(gpuidx):
    """
    Get GPU information

    Parameters
    ----------
    gpuidx : int
        GPU index

    Returns
    -------
    dict :
        GPU information in dictionary
    """
    import subprocess

    out_dict = {}
    try:
        sp = subprocess.Popen(
            ['nvidia-smi', '-q', '-i', str(gpuidx), '-d', 'MEMORY'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split('BAR1', 1)[0].split('\n')
        for item in out_list:
            try:
                key, val = item.split(':')
                key, val = key.strip(), val.strip()
                out_dict[key] = val
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
    return out_dict


def getfreegpumem(gpuidx):
    """
    Get free GPU memory

    Parameters
    ----------
    gpuidx : int
        GPU index

    Returns
    -------
    int :
        Free memory size
    """
    info = gpuinfo(gpuidx)
    if len(info) > 0:
        return int(info['Free'].replace('MiB', '').strip())
    else:
        return -1


def getbestgpu(log_level=0):
    """
    Get the GPU index with the largest free memory

    Parameters
    ----------
    log_level : int
        level to print message; 0: print everything, 1: only print selected device, 2: print nothing

    Returns
    -------
    int :
        The GPU index with the largest free memory
    """
    if torch.cuda.is_available():
        freememlist = []
        for gpuidx in range(torch.cuda.device_count()):
            freemem = getfreegpumem(gpuidx)
            freememlist.append(freemem)
            if log_level == 0:
                print('GPU device %d has %d MiB left.' % (gpuidx, freemem))
        idbest = freememlist.index(max(freememlist))
        if log_level < 2:
            print('--> GPU device %d was chosen' % idbest)
        return idbest
    else:
        if log_level < 2:
            print('No Nvidia GPU was detected.')
        return -1
