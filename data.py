class Data_Preprocessing():
    def Word_Embeding(data, min_count=1, sg=1, window=5, size=300, iter=200, stopwords=True):
        import gensim
        from tqdm import tqdm
        import numpy as np
        X = [i for i in data]
        model = gensim.models.Word2Vec(X, min_count=min_count, sg=sg,
                                       window=window,vector_size=size)
        if stopwords == True:
            def StopWordsList(filepath):
                stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
                return stopwords

            ting = StopWordsList('baidu_stopwords.txt')

            def trainW2C(s):
                words = str(s).lower()
                words = [w for w in words if not w in ting]
                M = []
                for w in words:
                    try:
                        M.append(model[w])
                    except:
                        continue
                M = np.array(M)
                v = M.sum(axis=0)
                if type(v) != np.ndarray:
                    return np.zeros(300)
                return v / np.sqrt((v ** 2).sum())

            word_vec = [trainW2C(x) for x in tqdm(X)]
            return word_vec
        else:
            def trainW2C(s):
                words = str(s).lower()
                M = []
                for w in words:
                    try:
                        # M.append(embeddings_index[w])
                        M.append(model[w])
                    except:
                        continue
                M = np.array(M)
                v = M.sum(axis=0)
                if type(v) != np.ndarray:
                    return np.zeros(300)
                return v / np.sqrt((v ** 2).sum())

            word_vec = [trainW2C(x) for x in tqdm(X)]
            return word_vec


class Assessment():
    def KFold_Assessment(model, train_vec, y_train, mode_name, cv=10):
        from sklearn.model_selection import cross_val_score
        scores_clf_svc_cv = cross_val_score(model, train_vec, y_train, cv=cv)
        print(mode_name + " Accuracy: %0.2f (+/- %0.2f)"
              % (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))

    def DNN_Loss_and_Accuracy(model):
        import matplotlib.pyplot as plt
        history_dict = model.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss', color="orange")
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        history_dict = model.history
        loss = history_dict['accuracy']
        val_loss = history_dict['val_accuracy']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training accuracy')
        plt.plot(epochs, val_loss, 'b', label='Validation accuracy', color="orange")
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.show()

    def Predict(modle, test_vec):
        test_predict = modle.predict(test_vec)
        return test_predict

    def F1(true_y, per_y, average='binary'):
        from sklearn.metrics import f1_score
        score = f1_score(true_y, per_y, average='binary')
        return score


