import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog


# Funcao para criar e treinar o modelo
def treinar_modelo():
    global modelo, X_test, y_test  

    # Simulacao de dados
    data = {
        'temperatura': np.random.uniform(15, 35, 1000),
        # 0 = Domingo, 6 = Sábado
        'dia_da_semana': np.random.randint(0, 7, 1000),
        'hora_do_dia': np.random.randint(0, 24, 1000),
        'consumo_kwh': np.random.uniform(0.5, 3.5, 1000) 
    }
    df = pd.DataFrame(data)

    # Pre-processamento dos dados
    df['temperatura'] = (df['temperatura'] - df['temperatura'].min()) / \
        (df['temperatura'].max() - df['temperatura'].min())
    df['hora_do_dia'] = df['hora_do_dia'] / 24  # Escalar de 0 a 1

    # Separacao em treino e teste
    X = df[['temperatura', 'dia_da_semana', 'hora_do_dia']]
    y = df['consumo_kwh']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Treinamento do modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Metricas de avaliacao
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Exibicao de metricas
    messagebox.showinfo("Treinamento Concluído",
                        f"Modelo treinado com sucesso!\n\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}")


# Função para prever o consumo de energia
def prever_consumo():
    try:
        # inputs
        temperatura = float(simpledialog.askstring(
            "Input", "Digite a temperatura atual (°C):"))
        dia_da_semana = int(simpledialog.askstring(
            "Input", "Digite o dia da semana (0=Domingo, 6=Sábado):"))
        hora_do_dia = float(simpledialog.askstring(
            "Input", "Digite a hora do dia (0-24):"))

        
        # Normalizar para a escala do modelo
        temperatura = (temperatura - 15) / (35 - 15)
        hora_do_dia = hora_do_dia / 24  

        # Predicao
        entrada = np.array([[temperatura, dia_da_semana, hora_do_dia]])
        previsao = modelo.predict(entrada)[0]

        # Exibicao do resultado
        messagebox.showinfo("Resultado", f"Previsão de Consumo de Energia:\n{
                            previsao:.2f} kWh")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao prever o consumo: {e}")



app = tk.Tk()
app.title("Previsão de Consumo de Energia Residencial")
app.geometry("400x200")


titulo = tk.Label(app, text="Previsão de Consumo de Energia",
                  font=("Arial", 14), pady=10)
titulo.pack()

# Botões
btn_treinar = tk.Button(app, text="Treinar Modelo",
                        font=("Arial", 12), command=treinar_modelo)
btn_treinar.pack(pady=10)

btn_prever = tk.Button(app, text="Prever Consumo",
                       font=("Arial", 12), command=prever_consumo)
btn_prever.pack(pady=10)


app.mainloop()
