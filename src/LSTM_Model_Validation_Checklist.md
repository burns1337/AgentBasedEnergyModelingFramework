
# Validierung eines LSTM-Modells

- [ ] **Aufteilung der Daten in Trainings-, Validierungs- und Testdaten**
  - [ ] Trainingsdaten (60-70%)
  - [ ] Validierungsdaten (10-20%)
  - [ ] Testdaten (10-20%)
  - [ ] Zeitliche Aufteilung beachten (ältere Daten für Training, neuere für Validierung und Test)

- [ ] **Metriken zur Bewertung**
  - [ ] Mean Absolute Error (MAE)
  - [ ] Mean Squared Error (MSE)
  - [ ] Root Mean Squared Error (RMSE)
  - [ ] Mean Absolute Percentage Error (MAPE)
  - [ ] Symmetric Mean Absolute Percentage Error (sMAPE)
  - [ ] Dokumentation der Metrikauswahl

- [ ] **Cross-Validation für Zeitreihen (Time-Series Cross-Validation)**
  - [ ] Rolling Window Cross-Validation anwenden
  - [ ] Stabilität der Vorhersagen über verschiedene Zeiträume messen

- [ ] **Benchmarking und Modellvergleich**
  - [ ] Vergleich mit Naiven Modellen (z.B. Persistence Model)
  - [ ] Vergleich mit exponentieller Glättung oder Mittelwertmodellen
  - [ ] Vergleich mit anderen Machine-Learning-Modellen (z.B. Random Forest)

- [ ] **Statistische Signifikanztests**
  - [ ] Diebold-Mariano-Test
  - [ ] Paired t-test oder Wilcoxon-Test (je nach Normalverteilung der Fehler)

- [ ] **Visuelle Analyse**
  - [ ] Plot der tatsächlichen vs. vorhergesagten Werte
  - [ ] Residual-Analyse
