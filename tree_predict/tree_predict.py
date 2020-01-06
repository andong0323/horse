#coding:utf-8
import xgboost as xgb
import argparse
from io_util_model_analysis import generate_fea_vec, feature_vecs_rm_missing
from common.util import fea_vecs_to_sparse, get_leaf_node_val, get_path
from xgb_forest_parser import XgbForestParser
import numpy as np
import math
from config.util import *
from common.util import parse_norm_file
from common.io_util import gen_new_title

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def create_parser():
	# parse command line input
	parser = argparse.ArgumentParser(description='data format')
	parser.add_argument('--binary_model', '-bm', help='input model file')
	parser.add_argument('--txt_model', '-tm', help='input model file', required=True)
	parser.add_argument('--feature_file', '-f', help='input feature file', required=True)
	parser.add_argument('--feature_file_type', '-fea_type', help='input feature file type: svm or tab or json', required=True)
	parser.add_argument('--feature_map', '-fea_map', help='input feature map file', nargs='?')
	parser.add_argument('--matrix_file', '-m', help='embedding matrix file', nargs='?')
	parser.add_argument('--norm_file', '-n', help='norm file name', nargs='?')

	return parser

def main():
	args = create_parser().parse_args()
	binary_model_file = args.binary_model
	txt_model_file = args.txt_model
	feature_file = args.feature_file
	feature_file_type = args.feature_file_type
	if args.feature_map is not None:
		fea_map = args.feature_map
	else:
		fea_map = None

	fea_vecs = generate_fea_vec(feature_file_type, feature_file, fea_map)
	if args.matrix_file and args.norm_file:
		feature_name_list = get_feature_list(PROJ)
		embed_weights = np.loadtxt(args.matrix_file, delimiter=',')
		emb_feas = gen_new_title(embed_weights.shape[1])
		fea_max, fea_min = parse_norm_file(args.norm_file)
		for fea_vec in fea_vecs:
			fea_vec_array = np.array([fea_vec[fea] for fea in feature_name_list])
			fea_vec_array = (fea_vec_array - fea_min) / (fea_max - fea_min)
			where_are_NaNs = np.isnan(fea_vec_array)
	if binary_model_file:
		binarys = binary_model_file.split(',')
		pred_all = []
		for model_index in range(len(binarys)):
			binary = binarys[model_index]
			bst = xgb.Booster(model_file=binary)
			sparse_fea_vec = fea_vecs_to_sparse(fea_vecs, fea_map)
			dtest = xgb.DMatrix(sparse_fea_vec)
			ypred = bst.predict(dtest)
			ypred -= 0.5
			pred_all.append(ypred)
			print('model %s: pred %s' % (model_index, ypred))
		print("overall: %s" % np.array(pred_all).sum(axis=0))
	
	tree_parser = XgbForestParser(txt_model_file)
	tree_dict = tree_parser.get_tree_dict()
	for idx in range(len(fea_vecs)):
		fea_vec = fea_vecs[idx]
		final_val = 0
		for tree_num in range(len(tree_dict)):
			path = get_path(tree_dict[tree_num].get_root(), fea_vec)
			final_vec += get_leaf_node_val(path)
		print("origin: %s, sigmoid: %s" % (final_val, sigmoid(final_val)))

if __name__ == "__main__":
	main()
