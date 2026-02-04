import glob
"""
将 npz 文件名写入 txt 文件
"""
def write_name(npz_path,save_dir):
	#npz文件路径
	files = glob.glob(f'{npz_path}\*.npz')
	#txt文件路径
	f = open(save_dir,'w',encoding='utf-8')
	for i in files:
		name = i.split('\\')[-1]
		name = name[:-4]+'\n'
		f.write(name)

if __name__ == '__main__':
	save_path= r'D:\DataBase\cabel_train_datas\add_cable_datas\npz'
	txt_file_save_dir= r"D:\DataBase\cabel_train_datas\add_cable_datas\train.txt"
	write_name(save_path,txt_file_save_dir)