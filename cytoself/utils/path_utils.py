import re
from os.path import join, sep, basename


def split_path_at(path, n):
    # Split path at the n-th path element
    path_elements = split_path_all(path)
    return join(*path_elements[:n]), join(*path_elements[n:])


def extract_path_at(path, n):
    return basename(split_path_at(path, n)[0][:-1])


def split_path_all(path):
    if sep not in path:
        return path
    else:
        if path[0] != sep:
            path = sep + path
        if path[-1] != sep:
            path = path + sep
        path_elements = []
        sepind = [s.start() for s in re.finditer(sep, path)]
        for i in range(len(sepind) - 1):
            path_elements.append(path[sepind[i] + 1 : sepind[i + 1]])
        return path_elements
