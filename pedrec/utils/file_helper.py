import glob
import logging
import ntpath
import os
import re
from typing import List

logger = logging.getLogger(__name__)


def is_filename_matching_regex(filename: str, regex: str) -> bool:
    if regex is None:
        return True
    pattern = re.compile(regex)
    return pattern.match(filename) is not None


def get_img_paths_from_folder(img_dir: str) -> List[str]:
    file_types = ['.png', '.jpg', '.pgm', '.PNG', '.JPG', '.JPEG']
    img_paths = []
    for file_type in file_types:
        img_search_string = os.path.join(img_dir, "*" + file_type)
        img_paths.extend(glob.glob(img_search_string))
    return img_paths


def get_img_paths_from_folder_recursive(img_dir: str) -> List[str]:
    img_paths = get_img_paths_from_folder(img_dir)
    for sub_dir in get_immediate_subdirectories(img_dir):
        logger.info("Handle {}".format(sub_dir))
        img_paths += (get_img_paths_from_folder_recursive(sub_dir))
    return img_paths


def get_video_paths_from_folder(video: str) -> List[str]:
    file_types = ['.avi', '.mp4', '.mp4v', '.mov', '.mkv']
    vid_paths = []
    for file_type in file_types:
        vid_search_string = os.path.join(video, "*" + file_type)
        vid_paths.extend(glob.glob(vid_search_string))
    return vid_paths


def get_video_paths_from_folder_recursive(video_dir: str) -> List[str]:
    vid_paths = get_video_paths_from_folder(video_dir)
    for sub_dir in get_immediate_subdirectories(video_dir):
        logger.info("Handle {}".format(sub_dir))
        vid_paths += (get_video_paths_from_folder_recursive(sub_dir))
    return vid_paths


def get_subdir_paths_recursive(directory: str) -> List[str]:
    output_dirs = []
    sub_dirs = get_immediate_subdirectories(directory)
    for sub_dir in sub_dirs:
        output_dirs.append(sub_dir)
        output_dirs.extend(get_subdir_paths_recursive(sub_dir))
    return output_dirs


def get_immediate_subdirectories(directory: str) -> List[str]:
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


def get_filenames_without_extension(file_dir: str) -> List[str]:
    filenames = []
    files = os.listdir(file_dir)
    for filename in files:
        filenames.append(get_filename_without_extension(filename))
    return filenames


def get_filename_from_path(file_path: str) -> str:
    return ntpath.basename(file_path)


def get_filename_without_extension(filename: str) -> str:
    return os.path.splitext(filename)[0]


def get_extension(filename: str) -> str:
    """
    Returns the extension from a filename / path
    :param filename: filename or file path to the file which extension is required
    :return: The files extension (with .)
    """
    return os.path.splitext(filename)[1]


def get_create_path(path: str) -> str:
    """
    Returns the given path, if it doesn't exist it creates it on file system
    :param path: The requested path
    :return: The requested path (which was created on file system if it doesn't exists)
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_pickle_filename_from_file_path(file_path: str) -> str:
    """
    Removes the extension of the given file path and replaces it with .pkl
    :param file_path: The path to the original file
    :return: The path to the pickle file which has the same name, but a .pkl extension
    """
    return os.path.join(os.path.abspath(file_path), get_filename_without_extension(file_path) + ".pkl")


def get_autoincremented_filepath(file_path: str, fill_zeros: int = 5) -> str:
    file_path = os.path.expanduser(file_path)

    root, ext = os.path.splitext(os.path.expanduser(file_path))
    file_dir_path = os.path.dirname(root)
    filename = os.path.basename(root)
    index_str = "0"
    if fill_zeros and fill_zeros > 0:
        index_str = index_str.zfill(fill_zeros)
    candidate = "{}_{}{}".format(filename, index_str, ext)
    index = 0
    ls = set(os.listdir(file_dir_path))
    while candidate in ls:
        index_str = str(index)
        if fill_zeros and fill_zeros > 0:
            index_str = index_str.zfill(fill_zeros)
        candidate = "{}_{}{}".format(filename, index_str, ext)
        index += 1
    return os.path.join(file_dir_path, candidate)


def get_last_dir_name(dir_path: str) -> str:
    """
    Returns the last directory name in a path, e.g. /a/b/color -> color
    :param dir_path:
    :return:
    """
    return os.path.basename(os.path.normpath(dir_path))


def batch_rename_files_to_index(files_dir: str, sorted_file_names: List[str] = None, zfill: int = 6):
    """
    Renames all files from a sorted list (which must all be located in files_dir), to index.extension.
    e.g.
    a.jpg, b.jpg -> 000000.jpg, 000001.jpg
    """
    if sorted_file_names is None:
        sorted_file_names = sorted(get_img_paths_from_folder(files_dir))
    for file_num, file_name in enumerate(sorted_file_names):
        ext = get_extension(file_name)
        file_name_new = "{0}{1}".format(str(file_num).zfill(zfill), ext)
        os.rename(os.path.join(files_dir, file_name), os.path.join(files_dir, file_name_new))


def trim_end(string_in: str, ending: str):
    """
    Trims the end of a string when it's present.
    :param string_in: The full string
    :param ending: The part which should be removed from the end
    :return: The string with ending removed.
    """
    if string_in.endswith(ending):
        return string_in[:-len(ending)]
    return string_in
