import pandas as pd
import numpy as np
import os
import util
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

RED = (231, 76, 60)
GREEN = (50, 255, 126)
GRAY = (149, 165, 166)
WHITE = (255, 255, 255)


def getDataByYear(year, mask = None): # 获得某一年的某几列的数据
    if type(year) == int:
        year = [year]
    if type(mask) == str:
        mask = [mask]
    return data[data['year'].isin(year)][mask] if mask != None else data[data['year'].isin(year)]

def genPic(year, data, dir_name="pic"): # 创建单张图片
    if not os.path.exists(dir_name):
        os.makedirs('./'+dir_name)

    data_year = data[data['year'] == year]
    im = Image.new("RGBA", ((x_max-x_min)//30+1, (y_max-y_min)//30+1))
    for row in data_year.iterrows():
        row = row[1]
        x,y = row['x.coord'], row['y.coord']
        kill = row['bb.kill']
        im.putpixel(((x-x_min)//30,(y-y_min)//30), RED if kill else GREEN)

    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im.info['transparency'] = 255
    im.save('./'+dir_name+'/{}.png'.format(year))
    return im

def genPredictPics(model, year, mask, data, verbose = True, use_neighbor = True, M = 5):
    if type(year) == int:
        year = [year]
    X_train, y_train, X_test, y_test = splitTrainingDataByYear(year, mask, use_neighbor, M)
    model = trainByModel(model, X_train, y_train)
    prediction = predictAndMeasureByModel(model, X_test, y_test)[1]
    draw_data = getDataByYear(year, mask=['x.coord', 'y.coord', 'year', 'bb.kill'])
    draw_data['bb.kill'] = prediction
    for y in year:
        genPic(y, draw_data, "pred")
    

def genGif(data, exist=False): # 创建所有图片并合成为gif
    imgs= []
    years = data['year']
    if exist:
        path = os.getcwd()+"/pic/"
        file_names = os.listdir(path)
        for file_name in file_names:
            im = Image.open(path + file_name, "r")
            im.info['transparency'] = 255
            imgs.append(im)
    else:
        for year in range(years.min(), years.max()+1):
            imgs.append(genPic(year, data))
    imgs[0].save("./pic/change.gif", 'GIF', save_all=True, append_images=imgs, duration=200, loop=0, disposal=2)

def genTrainingDataByYear(year, mask, use_neighbor, M=5): # 获得格式化的X与y便于进行训练
    _mask = mask + ['bb.kill']
    data_year = getDataByYear(year, _mask)
    X,y =  data_year[mask], data_year['bb.kill']
    if(use_neighbor):
        adjResults = util.genResultByYears(year, M)
        X = np.concatenate([X, adjResults[:,1:]], axis = 1)
    X[np.isnan(X)] = 0
    return X,y

def splitTrainingDataByYear(year, mask, use_neighbor, M=5): # 传入要预测的年份，将剩下的年份的数据切割成训练集和测试集
    years = data['year'].unique()
    if type(year) == int:
        year = [year]
    predict_years = year
    train_years = set(years) - set(predict_years)
    X_train, y_train = genTrainingDataByYear(train_years, mask, use_neighbor, M)
    X_test, y_test = genTrainingDataByYear(predict_years, mask, use_neighbor, M)
    return X_train, y_train, X_test, y_test

def trainByModel(model, X_train, y_train): # 传入选择的训练模型进行训练
    if(type(model) != XGBClassifier):
        model.fit(X_train, y_train, [8 if i == 1 else 1 for i in y_train])
    else:
        model.fit(X_train, y_train)
    return model

def predictAndMeasureByModel(model, X_test, y_test, verbose = True): # 传入测试数据进行预测并评估效果
    y_pred = model.predict(X_test)
    prediction = np.array([round(value) for value in y_pred])
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

    metrics_result = {
        'Accuracy':metrics.accuracy_score(y_test, prediction) * 100.0,
        'Recall':metrics.recall_score(y_test, prediction) * 100.0,
        'Precision':metrics.precision_score(y_test, prediction) * 100.0,
        'F1-score':metrics.f1_score(y_test, prediction) * 100.0,
        'AUC':metrics.auc(fpr, tpr) * 100.0,
    }

    if verbose:
        print("Accuracy: %.2f%%"%(metrics_result['Accuracy']))
        print('Recall: %.2f%%'%(metrics_result['Recall']))
        print('Precision: %.2f%%'%(metrics_result['Precision']))
        print('F1-score: %.2f%%'%(metrics_result['F1-score']))
        print('AUC: %.2f%%'%(metrics_result['AUC']))
    
    return metrics_result, prediction

def trainAndMeasure(model, year, mask, use_neighbor = True, M = 5):
    X_train, y_train, X_test, y_test = splitTrainingDataByYear(year, mask, use_neighbor, M)
    model = trainByModel(model, X_train, y_train)
    predictAndMeasureByModel(model, X_test, y_test)
    return model

def searchForBestWeight(year, mask, use_neighbor = True, M = 5):
    X_train, y_train, X_test, y_test = splitTrainingDataByYear(year, mask, use_neighbor, M)
    for i in range(1,10): 
        start = time.perf_counter()
        model = trainByModel(XGBClassifier(scale_pos_weight=i), X_train, y_train)
        predictAndMeasureByModel(model, X_test, y_test)
        now = time.perf_counter()
        print("scale_pos_weight=%d train time: %f"%(i, now - start))
    
def searchForBestM(year, mask, scale_pos_weight=8):
    for i in range(0,4):
        M = 2 * i + 1
        X_train, y_train, X_test, y_test = splitTrainingDataByYear(year, mask, True, M)
        start = time.perf_counter()
        model = trainByModel(XGBClassifier(scale_pos_weight=scale_pos_weight), X_train, y_train)
        predictAndMeasureByModel(model, X_test, y_test)
        now = time.perf_counter()
        print("M=%d train time: %f"%(M, now - start))

def searchForBestModel(year, mask):
    X_train, y_train, X_test, y_test = splitTrainingDataByYear(year, mask, True, M=5)
    models_for_selection = [XGBClassifier(scale_pos_weight=8), DecisionTreeClassifier(), RandomForestClassifier()]
    models_name = ["XGBoost", " DecisionTree", "RandomForest"]
    for i in range(0, len(models_for_selection)):
        start = time.perf_counter()
        model = trainByModel(models_for_selection[i], X_train, y_train)
        predictAndMeasureByModel(model, X_test, y_test)
        now = time.perf_counter()
        
        print("%s train time: %f"%(models_name[i], now - start))

def trainAndPlotByShap(year, mask):
    X_train, y_train, X_test, y_test = (pd.DataFrame(i) for i in splitTrainingDataByYear(year, mask, False, M=5))
    X_train.columns = mask + [str(i) for i in range(X_train.shape[1]-len(mask))]
    model = trainByModel(XGBClassifier(scale_pos_weight=7), X_train, y_train)    
    fm.fontManager.addfont('Dengb.ttf')
    plt.rcParams['font.sans-serif'] = "DengXian"
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values[:,:X_train.shape[1]], X_train.iloc[:,:X_train.shape[1]], plot_type="bar", show=False, max_display=32, plot_size=(14,18))
    plt.title('不同特征重要性排序')
    plt.savefig('top.png')

if __name__ == '__main__':
    data = pd.read_csv("data.txt", sep = ' ')
    data_1990 = getDataByYear(1990)
    xy_pairs = data_1990[['x.coord', 'y.coord']].values
    x_min, x_max = xy_pairs[:,0].min(), xy_pairs[:,0].max()
    y_min, y_max = xy_pairs[:,1].min(), xy_pairs[:,1].max()

    year = [1993, 2007]
    mask = ['S3.outbreak','S3.pothost','S3.outbreak2','S3.pothost2','S2.bb.pathlength','S2.host.diameter','S3.temp.yr','S3.temp.gs','S3.prec.yr','S3.prec.gs','S3.rad.yr','S3.rad.gs','S3.wetness']
    
    
    model = XGBClassifier(scale_pos_weight=8)
    trainAndMeasure(model, year, mask)