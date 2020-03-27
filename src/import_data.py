import pandas as pd

def elapsedtime_to_sec(el):
    tab = el.split(":")
    return float(tab[0])*60+float(tab[1])

def load_first_data(dataFolderName = '../data/', drop_default = True):
	listVideoName =['deadline_cif/x264-results1.csv',
			'bridge_close_cif/x264-results1.csv',
			'720p50_parkrun_ter/x264-results1.csv',
			'akiyo_qcif/x264-results1.csv',
			'bridge_far_cif/x264-results1.csv',
			'sunflower_1080p25/x264-results1.csv',
			'sintel_trailer_2k_480p24/x264-results1.csv',
			'husky_cif/x264-results1.csv',
			'netflix/x264-results1.csv',
			'waterfall_cif/x264-results1.csv',
			'claire_qcif/x264-results1.csv',
			'FourPeople_1280x720_60/x264-results1.csv',
			'students_cif/x264-results1.csv',
			'mobile_sif/x264-results1.csv',
			'flower_sif/x264-results1.csv',
			'riverbed_1080p25/x264-results1.csv',
			'blue_sky_1080p25/x264-results1.csv',
			'football_cif/x264-results1.csv',
			'tractor_1080p25/x264-results1.csv',
			'football_cif_15fps/x264-results1.csv',
			'tennis_sif/x264-results1.csv',
			'ducks_take_off_420_720p50/x264-results1.csv',
			'ice_cif/x264-results1.csv',
			'crowd_run_1080p50/x264-results1.csv',
			'soccer_4cif/x264-results1.csv']

	# creation of the list of videos (for each video: x264 configurations + measurements)
	listVideo = []

	for vn in listVideoName:
	    listVideo.append(pd.read_csv(open(dataFolderName+vn,"r")))

        # drop the default configurations
	if drop_default:
	    for i in range(len(listVideo)):
                listVideo[i] = listVideo[i].query('configurationID<=1152')

	for i in range(len(listVideo)):
            listVideo[i]['elapsedtime'] = [*map(elapsedtime_to_sec, listVideo[i]['elapsedtime'])]

	print(listVideo[0].head())

	# test
	print("There are " + str(len(listVideo)) + " videos")
	assert(len(listVideoName) == len(listVideo))

	return listVideo

def load_data(dataFolderName = '../data/', drop_default = True):
	listVideoName =['deadline_cif',
			'bridge_close_cif',
			'720p50_parkrun_ter',
			'akiyo_qcif',
			'bridge_far_cif',
			'sunflower_1080p25',
			'sintel_trailer_2k_480p24',
			'husky_cif',
			'netflix',
			'waterfall_cif',
			'claire_qcif',
			'FourPeople_1280x720_60',
			'students_cif',
			'mobile_sif',
			'flower_sif',
			'riverbed_1080p25',
			'blue_sky_1080p25',
			'football_cif',
			'tractor_1080p25',
			'football_cif_15fps',
			'tennis_sif',
			'ducks_take_off_420_720p50',
			'ice_cif',
			'crowd_run_1080p50',
			'soccer_4cif']

	# creation of the list of videos (for each video: x264 configurations + measurements)
	listVideo = []

	for vn in listVideoName:
            listVideo.append(pd.read_csv(open(dataFolderName+vn+'/result_mean.csv',"r")))

        # drop the default configurations
	if drop_default:
	    for i in range(len(listVideo)):
                listVideo[i] = listVideo[i].query('configurationID<=1152')

	print(listVideo[0].head())

	# test
	print("There are " + str(len(listVideo)) + " videos")
	assert(len(listVideoName) == len(listVideo))

	return listVideo


