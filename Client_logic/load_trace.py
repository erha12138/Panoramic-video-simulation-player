import os


COOKED_TRACE_FOLDER = './traces/'
# COOKED_TRACE_FOLDER = './traces_belgium/'
# COOKED_TRACE_FOLDER = '../Network_Trace/'
# COOKED_TRACE_FOLDER = './traces_fcc/'
# COOKED_TRACE_FOLDER = './unfair/user1/'
# COOKED_TRACE_FOLDER = './fair/user11/'
# COOKED_TRACE_FOLDER = './traces_oboe/'

script_dir = os.path.dirname(os.path.abspath(__file__))
# E:\abr-fov-dev\FoV-ABR-master\traces

def load_trace(train):
    if train:
        cooked_trace_folder="E:\\abr-fov-dev\\FoV-ABR-master\\traces_in_range/"
    else:
        cooked_trace_folder="E:\\abr-fov-dev\\FoV-ABR-master\\traces_in_range/"
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names
