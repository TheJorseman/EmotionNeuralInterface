import os
import mne
# Data from https://physionet.org/content/ltrsvp/1.0.0/
# https://physionet.org/content/eegmmidb/1.0.0/
# https://physionet.org/content/eegmat/1.0.0/
folder = "../Datasets/eeg-signals-from-an-rsvp-task-1.0.0"
output_folder = "../Datasets/eeg-signals-from-an-rsvp-task-1.0.0/output"

#files_data['eeg-signals-from-an-rsvp-task-1.0.0']['6-Hz'][0].to_data_frame()
#filename = files_data['eeg-signals-from-an-rsvp-task-1.0.0']['6-Hz'][0]._filenames[0]

def get_id_type(filename):
    base = os.path.basename(filename).split("_")[2].split(".")[0]
    return int(base[:2]), base[-1]

def get_files(folder, ext):
    files = {'folder_name': folder, 'files': [], 'ext': ext}
    for file in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, file)):
            files['files'].append(get_files(os.path.join(folder, file), ext))
        if file.endswith(ext):
            files['files'].append(os.path.join(folder, file))
    return files

def get_data_from_files(files):
    result = {}
    for file in files['files']:
        if isinstance(file, dict):
            key = os.path.basename(file['folder_name'])
            result[key] = [mne.io.read_raw_edf(file, preload=True) for file in file['files']]
    return result

def save_data_to_csv(data, output_folder):
    for key, value in data.items():
        os.makedirs(os.path.join(output_folder, key), exist_ok=True)
        for item in data[key]:
            df = item.to_data_frame()
            df = df.drop(['Channel'], axis=1)
            filename = item._filenames[0]
            id,type = get_id_type(filename)
            output = os.path.join(output_folder, key, "rsvp_s_{}_{}.csv".format(id, type))
            df.to_csv(output, index=False)

files = get_files(folder, '.edf')
files_data = get_data_from_files(files)
save_data_to_csv(files_data, output_folder)
import pdb;pdb.set_trace()