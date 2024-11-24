import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import shap

# Importar o módulo Classificador
from Classificador import preparar_modelos

# Obter as variáveis necessárias
modelos_info = preparar_modelos()

if modelos_info is None:
    print("Erro ao preparar os modelos. Encerrando a interface gráfica.")
    exit()

# Extrair as variáveis do dicionário
df = modelos_info['df']
modelos_dict = modelos_info['modelos']
contextos = modelos_info['contextos']
classes = modelos_info['classes']
tipos_grafico = modelos_info['tipos_grafico']
colunas_relevantes = modelos_info['colunas_relevantes']
limpar_nomes_colunas = modelos_info['limpar_nomes_colunas']

class InterfaceGrafica:
    def __init__(self, root):
        self.root = root
        self.modelos = modelos_dict
        self.contextos = contextos
        self.classes = classes
        self.tipos_grafico = tipos_grafico

        self.root.title("Análise de Importância de Variáveis")
        self.root.geometry("1500x1000")

        # Criar widgets
        self.criar_widgets()

    def criar_widgets(self):

        # Frame para os seletores
        frame_seletores = tk.Frame(self.root)
        frame_seletores.pack(pady=10, padx=10, fill=tk.X)

        # Seção de Região
        regioes = sorted(list(set([
            regiao.strip() for regiao in self.contextos
            if regiao.startswith('Região ') and ' - ' not in regiao
        ])))
        regioes.insert(0, 'Base Total')
        self.combo_regiao = ttk.Combobox(frame_seletores, values=regioes, state="readonly", width=25)
        self.combo_regiao.set('Base Total')
        self.combo_regiao.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Seção de Período
        periodos = sorted(list(set([
            periodo.strip() for periodo in self.contextos
            if periodo.startswith('Período ') and ' - ' not in periodo
        ])))
        periodos.insert(0, 'All')
        self.combo_periodo = ttk.Combobox(frame_seletores, values=periodos, state="readonly", width=25)
        self.combo_periodo.set('All')
        self.combo_periodo.grid(row=0, column=3, padx=5, pady=5, sticky='w')

        # Seção de Classificador
        tk.Label(frame_seletores, text="Classificador:").grid(row=0, column=4, padx=5, pady=5, sticky='e')
        self.combo_classe = ttk.Combobox(frame_seletores, values=self.classes, state="readonly", width=15)
        self.combo_classe.set(self.classes[0])
        self.combo_classe.grid(row=0, column=5, padx=5, pady=5, sticky='w')

        # Seção de Tipo de Gráfico
        tk.Label(frame_seletores, text="Tipo de Gráfico:").grid(row=0, column=6, padx=5, pady=5, sticky='e')
        self.combo_tipo = ttk.Combobox(frame_seletores, values=self.tipos_grafico, state="readonly", width=20)
        self.combo_tipo.set(self.tipos_grafico[0])
        self.combo_tipo.grid(row=0, column=7, padx=5, pady=5, sticky='w')

        # Botão para gerar o gráfico
        tk.Button(frame_seletores, text="Gerar Gráfico", command=self.gerar_grafico).grid(row=0, column=8, padx=10, pady=5)

        # Frame para o gráfico
        self.frame_grafico = tk.Frame(self.root)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas do gráfico
        self.canvas = None

    def gerar_grafico(self):

        regiao = self.combo_regiao.get()
        periodo = self.combo_periodo.get()
        classe = self.combo_classe.get()
        tipo_grafico = self.combo_tipo.get()

        df_filtrado = df.copy()

        # Limpar 'Região ' e 'Período ' dos valores selecionados
        if regiao != 'Base Total':
            regiao_clean = regiao.replace('Região ', '').strip()
            df_filtrado = df_filtrado[df_filtrado['regiao'].astype(str).str.lower() == regiao_clean.lower()]
        else:
            regiao_clean = None

        if periodo != 'All':
            periodo_clean = periodo.replace('Período ', '').strip()
            df_filtrado = df_filtrado[df_filtrado['periodo_de_pesquisa'] == periodo_clean]
        else:
            periodo_clean = None

        # Verificar se há dados após os filtros
        if df_filtrado.empty:
            messagebox.showwarning("Aviso", "Nenhum dado disponível para as seleções feitas.")
            return

        # Determinar o contexto para selecionar o modelo
        if regiao_clean and periodo_clean:
            contexto = f'Região {regiao_clean} - Período {periodo_clean}'
        elif regiao_clean:
            contexto = f'Região {regiao_clean}'
        elif periodo_clean:
            contexto = f'Período {periodo_clean}'
        else:
            contexto = 'Base Total'

        # Tentar encontrar o modelo correspondente
        if (contexto, classe) not in self.modelos:
            # Tentar alternativas se o modelo não existir
            if regiao_clean and (f'Região {regiao_clean}', classe) in self.modelos:
                contexto = f'Região {regiao_clean}'
            elif periodo_clean and (f'Período {periodo_clean}', classe) in self.modelos:
                contexto = f'Período {periodo_clean}'
            elif ('Base Total', classe) in self.modelos:
                contexto = 'Base Total'
            else:
                messagebox.showwarning("Aviso", f"Modelo para '{classe}' não disponível.")
                return

        # Obter o modelo correspondente
        modelo_info = self.modelos.get((contexto, classe))

        # Verificar existência do modelo
        if modelo_info is None:
            messagebox.showwarning("Aviso", f"Modelo para '{classe}' em '{contexto}' não disponível.")
            return

        modelo = modelo_info['modelo']

        if tipo_grafico == 'Importância do Modelo':
            # Obter as top variáveis e importâncias
            top_vars_info = modelo_info['top_vars']
            if not top_vars_info:
                messagebox.showwarning("Aviso", f"Não há variáveis de importância para o modelo '{classe}' em '{contexto}'.")
                return

            # Extrair top variáveis e importâncias
            top_vars, importances_percent = zip(*top_vars_info)
            # Formatar nomes das variáveis
            top_vars_formatadas = [self.formatar_nome_variavel(var) for var in top_vars]
            titulo = f"Importância das Variáveis - {classe.capitalize()} - {contexto}"
            # Plotar o gráfico de importâncias
            self.plotar_grafico_importancia(top_vars_formatadas, importances_percent, titulo)
        else:
            # Valores SHAP
            X_shap, shap_values, legend_labels = self.obter_valores_shap(modelo, df_filtrado)
            if X_shap is None or shap_values is None:
                messagebox.showwarning("Aviso", "Não foi possível gerar valores SHAP para este modelo.")
                return
            titulo = f"Resumo dos Valores SHAP - {classe.capitalize()} - {contexto}"
            # Plotar o gráfico SHAP
            self.plotar_grafico_shap(X_shap, shap_values, legend_labels, titulo)

    def plotar_grafico_importancia(self, top_vars, importances, titulo):

        # Limpar o frame do gráfico
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()

        # Criar a figura com um tamanho adequado
        fig = plt.Figure(figsize=(16, 8), dpi=100)
        ax = fig.add_subplot(111)

        # Inverter a ordem para que o mais importante fique no topo
        importances_reversed = importances[::-1]
        top_vars_reversed = top_vars[::-1]
        indices = np.arange(len(top_vars))

        # Gerar cores únicas para cada barra
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_vars_reversed)))

        # Criar gráfico de barras horizontais
        bars = ax.barh(indices, importances_reversed, color=colors)
        ax.set_title(titulo, fontsize=18)
        ax.set_xlabel('Importância (%)', fontsize=14)
        ax.set_yticks(indices)
        ax.set_yticklabels([f"{i + 1}" for i in indices])

        ax.invert_yaxis()

        # Formatar o eixo x para mostrar porcentagens
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1f}%'))

        # Criar legenda mapeando números às variáveis
        from matplotlib.patches import Patch

        legend_labels = [f"{i + 1}. {var}" for i, var in enumerate(top_vars_reversed)]
        legend_handles = [Patch(facecolor=colors[i], edgecolor='black') for i in range(len(colors))]

        # Adicionar a legenda abaixo do gráfico
        ax.legend(
            legend_handles,
            legend_labels,
            title="Top 10 Variáveis",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=1,
            fontsize=10,
            title_fontsize=12,
            frameon=True,
            borderaxespad=0.0
        )

        # Ajustar layout para acomodar a legenda
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.35)

        # Embutir o gráfico no frame
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def plotar_grafico_shap(self, X, shap_values, legend_labels, titulo):

        # Limpar o frame do gráfico
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()

        # Fechar todas as figuras para evitar acúmulo
        plt.close('all')

        # Criar uma nova figura
        plt.figure(figsize=(12, 8), dpi=100)

        # Gerar o gráfico SHAP summary plot
        try:
            shap.summary_plot(
                shap_values,
                X,
                plot_size=(12, 8),
                show=False,
                color='viridis'
            )
            plt.title(titulo, fontsize=16)

            # Obter o objeto Axis atual
            ax = plt.gca()

            # Atualizar os labels do eixo Y para números
            ax.set_yticklabels([f"{i+1}" for i in range(len(X.columns))])

            # Criar legenda mapeando números às variáveis
            from matplotlib.patches import Patch

            # Criar patches brancos para a legenda
            legend_patches = [Patch(facecolor='white', edgecolor='black')] * len(legend_labels)

            # Adicionar a legenda abaixo do gráfico
            plt.legend(
                legend_patches,
                legend_labels,
                title="Top 10 Variáveis",
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=1,
                fontsize=10,
                title_fontsize=12,
                frameon=True,
                borderaxespad=0.0
            )

            # Ajustar layout para acomodar a legenda
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.35)

            # Obter o objeto Figure atual
            fig = plt.gcf()

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico SHAP: {e}")
            return

        # Embutir o gráfico no frame
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def obter_valores_shap(self, modelo, df_contexto):

        # Preparar dados para SHAP
        try:
            X = df_contexto[colunas_relevantes]
            X = limpar_nomes_colunas(X)
            feature_names = modelo.get_booster().feature_names
            if feature_names is None:
                messagebox.showerror("Erro", "O modelo não possui nomes de features.")
                return None, None, None
            X = X[feature_names]
        except KeyError as e:
            messagebox.showerror("Erro", f"Erro ao selecionar colunas para SHAP: {e}")
            return None, None, None
        except Exception as e:
            messagebox.showerror("Erro", f"Erro inesperado: {e}")
            return None, None, None

        # Preencher valores nulos com zero
        X = X.fillna(0)

        # Verificar se todas as colunas necessárias estão presentes
        missing_features = [feat for feat in feature_names if feat not in X.columns]
        if missing_features:
            messagebox.showerror("Erro", f"Faltam as seguintes colunas no DataFrame para SHAP: {missing_features}")
            return None, None, None

        # Criar explicador e calcular valores SHAP
        try:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular valores SHAP: {e}")
            return None, None, None

        # Se shap_values é uma lista (multiclass), selecionar a classe positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calcular os valores SHAP médios absolutos
        shap_vals_abs = np.abs(shap_values).mean(axis=0)

        # Obter os índices dos top 10 features
        top_indices = np.argsort(shap_vals_abs)[::-1][:10]

        # Selecionar as top features
        X_top = X.iloc[:, top_indices]
        shap_values_top = shap_values[:, top_indices]

        # Formatar e numerar os nomes das features
        legend_labels = []
        for idx, col in enumerate(X_top.columns):
            formatted_name = self.formatar_nome_variavel(col)
            legend_label = f"{idx + 1}. {formatted_name}"
            legend_labels.append(legend_label)

        # Renomear as colunas em X_top para números de 1 a 10
        X_top.columns = [str(idx + 1) for idx in range(len(X_top.columns))]

        return X_top, shap_values_top, legend_labels

    def formatar_nome_variavel(self, nome_variavel):

        # Remover o sufixo '_csat' se presente
        if nome_variavel.endswith('_csat'):
            nome_variavel = nome_variavel[:-5]
        # Substituir underscores e barras por espaços
        nome_variavel = nome_variavel.replace('_', ' ').replace('/', ' ')
        # Capitalizar as palavras
        nome_variavel = nome_variavel.title()
        # Opcional: Abreviar termos comuns para melhorar a legibilidade
        abreviacoes = {
            "Qualidade": "Qual.",
            "Confiabilidade": "Conf.",
            "Desempenho": "Desemp.",
            "Produto": "Prod.",
            "Aparencia": "Apar.",
            "Comodidade": "Comod.",
            "Ergonomia": "Ergo.",
            "Caracteristicas" : "Carac.",
        }
        for palavra, abreviacao in abreviacoes.items():
            nome_variavel = nome_variavel.replace(palavra, abreviacao)
        return nome_variavel

# Executar a interface gráfica
if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceGrafica(root)
    root.mainloop()
