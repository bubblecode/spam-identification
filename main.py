from spam_utils import load_data, tfidf_dataset, draw_loss_acc, draw_PR, draw_ROC, draw_confu, print_acc_err, print_PRF1_score
from spam_with_LG import LG
from spam_with_NB import bernoulli_NB
from spam_with_SVM import SVM

if __name__ == '__main__':
    """
    200,  500,  500
    3345, 1114, 1114
    """
    train_data,train_label,val_data,val_label,test_data,test_label = load_data(3345, 1114, 1114)

    train_data_idf, val_data_idf,_ = tfidf_dataset(train_data, val_data)

    train_data_idf = train_data_idf.todense()
    val_data_idf   = val_data_idf.todense()

    print('SVM begin ...')
    pred_svm = SVM(train_data_idf, train_label, val_data_idf)
    print('svm ok')
    print('LG begin ...')
    pred_lg, label_lg = LG(train_data_idf, train_label, val_data_idf, val_label)
    print('LG ok')
    print('bNB begin ...')
    pred_nb = bernoulli_NB(train_data_idf, train_label, val_data_idf)
    print('bNB ok')


    print_acc_err(pred_svm, val_label, name='SVM')
    print_acc_err(pred_lg,  label_lg,  name='LG')
    print_acc_err(pred_nb,  val_label, name='NB')
    
    print_PRF1_score(pred_svm, val_label, name='SVM')
    print_PRF1_score(pred_lg,  label_lg,  name='LG')
    print_PRF1_score(pred_nb,  val_label, name='NB')

    draw_PR(pred_svm, val_label, name="SVM")
    draw_PR(pred_lg, label_lg,   name="LG")
    draw_PR(pred_nb, val_label,  name="NB")
    
    draw_ROC(pred_svm, val_label, name="SVM")
    draw_ROC(pred_lg, label_lg,   name="LG")
    draw_ROC(pred_nb, val_label,  name="NB")

    draw_confu(pred_svm, val_label, name="SVM")
    draw_confu(pred_lg, label_lg,   name="LG")
    draw_confu(pred_nb, val_label,  name="NB")
