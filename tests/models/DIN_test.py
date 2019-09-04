import numpy as np

from deepctr.models.din import DIN
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_fixlen_feature_names,get_varlen_feature_names
from ..utils import check_model



def get_xy_fd(hash_flag=False):


    feature_columns = [SparseFeat('user',3),SparseFeat(
        'gender', 2), SparseFeat('item', 3 + 1), SparseFeat('item_gender', 2 + 1),DenseFeat('score', 1)]
    feature_columns += [VarLenSparseFeat('hist_item',3+1, maxlen=4, embedding_name='item'),
                        VarLenSparseFeat('hist_item_gender',3+1, maxlen=4, embedding_name='item_gender')]

    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score}

    feature_names = get_fixlen_feature_names(feature_columns)
    varlen_feature_names = get_varlen_feature_names(feature_columns)
    x = [feature_dict[name] for name in feature_names] + [feature_dict[name] for name in varlen_feature_names]


    y = [1, 0, 1]
    return x, y, feature_columns, behavior_feature_list


#@pytest.mark.xfail(reason="There is a bug when save model use Dice")
#@pytest.mark.skip(reason="misunderstood the API")


def test_DIN():
    model_name = "DIN"

    x, y, feature_columns, behavior_feature_list = get_xy_fd(True)

    model = DIN(feature_columns, behavior_feature_list, embedding_size=8, hist_len_max=4, dnn_hidden_units=[4, 4, 4],
                dnn_dropout=0.5)

    check_model(model,model_name,x,y)


if __name__ == "__main__":
    pass
