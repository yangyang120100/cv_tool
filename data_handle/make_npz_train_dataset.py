import glob
import argparse
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
	parser = argparse.ArgumentParser()
	parser.add_argument('--npz_path', type=str,
						default=r'D:\DataBase\cabel_train_datas\npz', help='npz文件路径')
	parser.add_argument('--txt_file_save_dir', type=str,
						default=r"D:\DataBase\cabel_train_datas\train.txt", help='输出txt路径')
	args = parser.parse_args()
	write_name(args.save_path,args.txt_file_save_dir)