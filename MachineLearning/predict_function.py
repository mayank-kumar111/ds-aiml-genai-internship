from sklearn.metrics import r2_score, mean_absolute_error


def predict(models, X_train, X_test, y_train, y_test):

    for name, model in models:

        print("=" * 60)
        print(f"Model: {name}")
        print("=" * 60)

        # Train
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"R2 Score: {r2:.4f}")
        print(f"MAE: {mae:.2f}")
        print()