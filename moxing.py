class Model():

    def LogisticRegression(train_vec, y_train, model_name=None):
        from sklearn import linear_model

        if model_name != "CV":
            clf = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
            return clf.fit(train_vec, y_train)
        else:
            clf = linear_model.LogisticRegressionCV(solver='lbfgs', multi_class='multinomial')
            return clf.fit(train_vec, y_train)

    def SVM(train_vec, y_train, kernel, C=1.2):
        from sklearn.svm import SVC
        clf = SVC(C=C, kernel=kernel)
        return clf.fit(train_vec, y_train)

    def XGB(train_vec, y_train,
            max_depth=7, n_estimators=200,
            colsample_bytree=0.8, subsample=0.8,
            nthread=10, learning_rate=0.1,
            verbosity=0):
        import xgboost as xgb
        clf = xgb.XGBClassifier(max_depth=max_depth,
                                n_estimators=n_estimators,
                                colsample_bytree=colsample_bytree,
                                subsample=subsample,
                                nthread=nthread,
                                learning_rate=learning_rate,
                                verbosity=verbosity)
        return clf.fit(train_vec, y_train)

    def RandomForestClassifier(train_vec, y_train, n_estimators=200, max_depth=7):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        return clf.fit(train_vec, y_train)

    def Bayes(train_vec, y_train):
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
        return clf.fit(train_vec, y_train, )

    def BGRU(train_x, train_y, input_shape, dev_x, dev_y, epochs=100, batch_size=128, optimizer='adam', ):
        from keras import models
        from keras import layers
        from keras.layers import Dense, Dropout, Embedding, GRU, Bidirectional
        model = models.Sequential()
        model.add(Dense(200, activation='sigmoid', input_shape=input_shape))
        #        model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_x,
                            train_y,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(dev_x, dev_y))
        return history, model

