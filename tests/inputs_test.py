from deepctr.models import DeepFM
from deepctr.inputs import DenseFeat, SparseFeat, get_input_feature_names, create_embedding_matrix, embedding_lookup, \
    build_input_layer_features, build_emd_layer_from_feature_columns
import numpy as np

feature_columns = [SparseFeat('user_id', 4, ), SparseFeat('item_id', 5, ), DenseFeat("pic_vec", 5)]


def test_create_embedding_matrix():
    fixlen_feature_names = get_input_feature_names(feature_columns)

    user_id = np.array([[1], [0], [1]])
    item_id = np.array([[3], [2], [1]])
    pic_vec = np.array([[0.1, 0.5, 0.4, 0.3, 0.2], [0.1, 0.5, 0.4, 0.3, 0.2], [0.1, 0.5, 0.4, 0.3, 0.2]])
    label = np.array([1, 0, 1])

    embedding_dict = create_embedding_matrix(feature_columns, l2_reg=1e-5, init_std=0.0001, seed=1024, embedding_size=4)

    # sparse_embedding_list = embedding_lookup(
    #     embedding_dict, features, feature_columns)


def test_build_input_features():
    input_layer_features = build_input_layer_features(feature_columns)
    print input_layer_features

    return input_layer_features


def test_input_from_feature_columns():
    input_features = test_build_input_features()
    inputs_list = list(input_features.values())

    sparse_embedding_list, dense_value_list = build_emd_layer_from_feature_columns(input_features, feature_columns,
                                                                                   l2_reg=1e-5, init_std=0.0001,
                                                                                   seed=1024,
                                                                                   embedding_size=4)
    print sparse_embedding_list
    print dense_value_list
