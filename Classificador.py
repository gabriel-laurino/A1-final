import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import unicodedata
import os

# Configuração de exibição de tabelas
pd.set_option('display.float_format', '{:.2f}'.format)

def classificar_nota(nota):
    if nota >= 9:
        return "promotor"
    elif nota >= 7:
        return "neutro"
    else:
        return "detrator"

def limpar_nomes_colunas(df):
    df = df.copy()
    # Remover caracteres especiais, incluindo ¿ e ?
    df.columns = df.columns.str.replace(r"[¿?\[\]<>\(\),\-]", "", regex=True)
    # Substituir espaços por underscores
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
    # Remover acentos
    df.columns = df.columns.map(lambda x: ''.join(
        c for c in unicodedata.normalize('NFKD', x) if not unicodedata.combining(c)))
    # Converter para minúsculas
    df.columns = df.columns.str.lower()
    return df

def limpar_periodo(periodo):
    if pd.isna(periodo):
        return periodo
    # Remover espaços extras e converter para minúsculas
    periodo = periodo.strip().lower()
    # Remover múltiplos espaços
    periodo = ' '.join(periodo.split())
    # Substituir ' a ' por ' A ' para capitalizar corretamente
    periodo = periodo.replace(' a ', ' A ')
    # Converter para título
    return periodo.title()

def preparar_modelos():
    # Passo 1: Carregar os dados
    try:
        df = pd.read_excel("Lista NPS Positivo_V4.xlsx")
    except FileNotFoundError:
        print("Erro: O arquivo 'Lista NPS Positivo_V4.xlsx' não foi encontrado.")
        return
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # Passo 2: Limpar os nomes das colunas
    df = limpar_nomes_colunas(df)

    # Passo 2.1: Remover colunas duplicadas
    duplicated_columns_initial = df.columns[df.columns.duplicated()]
    if len(duplicated_columns_initial) > 0:
        df = df.loc[:, ~df.columns.duplicated()]

    # Passo 3: Verificar se as colunas necessárias existem
    if not {'mercado', 'grupo_de_produto', 'nota'}.issubset(df.columns):
        print("\nErro: As colunas 'mercado', 'grupo_de_produto' e/ou 'nota' não existem no DataFrame.")
        return

    # Passo 4: Classificação da coluna 'nota'
    df['target'] = df['nota'].apply(classificar_nota)

    # Passo 5: Garantir que a coluna 'target' possui todas as categorias
    categorias_target = ['promotor', 'neutro', 'detrator']
    df['target'] = pd.Categorical(df['target'], categories=categorias_target)

    # Passo 6: Verificar contagens antes dos filtros
    print("\nContagem por categoria antes dos filtros:")
    print(df['target'].value_counts())

    # Passo 7: Normalizar as colunas de períodos usando a função de limpeza
    df['periodo_de_pesquisa'] = df['periodo_de_pesquisa'].astype(str).apply(limpar_periodo)
    df['periodo_de_pesquisa'] = df['periodo_de_pesquisa'].str.replace('.', '', regex=False).str.replace(',', '', regex=False)

    # Passo 8: Filtrar pelo mercado Brasil e Grupo 1
    df_filtered = df[
        (df['mercado'].str.lower().str.strip() == 'brasil') &
        (df['grupo_de_produto'].str.lower().str.strip() == 'grupo 1')
    ].copy()

    # Passo 9: Verificar contagens após os filtros
    if not df_filtered.empty and 'target' in df_filtered.columns:
        print("\nContagem por categoria após os filtros:")
        print(df_filtered['target'].value_counts())
    else:
        print("\nErro: DataFrame vazio após os filtros ou coluna 'target' não encontrada.")

    # Passo 10: Mapeamento de regiões
    regioes = {
        # Região Norte
        "AC": "norte", "AP": "norte", "AM": "norte",
        "PA": "norte", "RO": "norte", "RR": "norte", "TO": "norte",
        # Região Nordeste
        "AL": "nordeste", "BA": "nordeste", "CE": "nordeste",
        "MA": "nordeste", "PB": "nordeste", "PE": "nordeste",
        "PI": "nordeste", "RN": "nordeste", "SE": "nordeste",
        # Região Centro-Oeste
        "DF": "centro-oeste", "GO": "centro-oeste",
        "MT": "centro-oeste", "MS": "centro-oeste",
        # Região Sudeste
        "ES": "sudeste", "MG": "sudeste",
        "RJ": "sudeste", "SP": "sudeste",
        # Região Sul
        "PR": "sul", "RS": "sul", "SC": "sul"
    }

    if 'estado' in df_filtered.columns:
        df_filtered['estado'] = df_filtered['estado'].str.upper().str.strip()
        df_filtered['regiao'] = df_filtered['estado'].map(regioes)
    else:
        df_filtered['regiao'] = np.nan
        print("\nAviso: A coluna 'estado' não existe no DataFrame. A coluna 'regiao' será preenchida com NaN.")

    # Passo 11: Extrair 'safra' a partir de 'data_resposta'
    if 'data_resposta' in df_filtered.columns:
        df_filtered['safra'] = pd.to_datetime(df_filtered['data_resposta'], errors='coerce').dt.year
    else:
        df_filtered['safra'] = np.nan
        print("\nAviso: A coluna 'data_resposta' não existe no DataFrame. A coluna 'safra' será preenchida com NaN.")

    # Passo 12: Identificar as colunas de perguntas
    csat_columns = [col for col in df_filtered.columns if 'csat' in col.lower()]

    # Remover colunas 'csat' que têm todos os campos vazios
    empty_csat_columns = [col for col in csat_columns if df_filtered[col].isnull().all()]
    df_filtered = df_filtered.drop(columns=empty_csat_columns)

    # Remover colunas 'csat' que estão em espanhol
    spanish_patterns = ['like/dislike', 'multiple', 'input']
    spanish_csat_columns = [col for col in csat_columns if any(pattern in col for pattern in spanish_patterns)]
    df_filtered = df_filtered.drop(columns=spanish_csat_columns)

    # Remover colunas em espanhol adicionais manualmente
    manual_spanish_csat_columns = [
        'esta_satisfecho_con_la_calidad_del_producto_csat',
        'agora_considere_las_caracteristicas_especificas_de_su_modelo_mientras_evalua_su_desempeno._csat',
        'como_evalua_la_comodidad_y_la_ergonomia_de_su_modelo_considere_por_ejemplo_la_comodidad_de_los_asientos_la_visibilidad_de_la_cabina_la_comprensibilidad_y_la_disposicion_de_los_controles_csat',
        'p3_como_voce_avalia_a_qualidade_e_a_confiabilidade_do_seu_modelo_considere_o_acabamento_e_a_aparencia_da_sua_maquina_falhas_problemas_de_confiabilidade_ou_avarias_que_voce_possa_ter_tido_csat'
    ]
    df_filtered = df_filtered.drop(columns=manual_spanish_csat_columns, errors='ignore')

    # Remover colunas duplicadas após a limpeza
    duplicated_columns_after_spanish = df_filtered.columns[df_filtered.columns.duplicated()]
    if len(duplicated_columns_after_spanish) > 0:
        df_filtered = df_filtered.loc[:, ~df_filtered.columns.duplicated()]

    # Atualizar a lista de colunas 'csat' após remoções
    csat_columns = [col for col in df_filtered.columns if 'csat' in col.lower()]

    # Passo 16: Converter colunas 'csat' para numérico
    for col in csat_columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # Passo 17: Remover linhas onde a coluna 'nota' está ausente ou não é numérica
    df_filtered['nota'] = pd.to_numeric(df_filtered['nota'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['nota'])

    # Passo 18: Garantir que os valores nas colunas 'nota' e 'csat' estejam entre 0 e 10
    for col in ['nota'] + csat_columns:
        df_filtered.loc[(df_filtered[col] < 0) | (df_filtered[col] > 10), col] = pd.NA

    # Passo 19: Identificar as colunas de perguntas válidas
    question_columns = csat_columns.copy()
    valid_question_columns = []
    columns_to_drop = []
    max_nan_ratio = 1  # 100%

    for col in question_columns:
        nan_ratio = df_filtered[col].isnull().mean()
        if nan_ratio > max_nan_ratio:
            columns_to_drop.append(col)
            continue
        numeric_responses = df_filtered[col].dropna()
        if not numeric_responses.empty and numeric_responses.between(0, 10).all():
            valid_question_columns.append(col)
        else:
            columns_to_drop.append(col)

    # Passo 21: Selecionar as colunas válidas
    df_final = df_filtered.drop(columns=columns_to_drop)

    # Passo 22: Calcular a correlação de Spearman entre 'nota' e as colunas de perguntas
    if valid_question_columns:
        correlacoes = df_final[['nota'] + valid_question_columns].corr(method='spearman', min_periods=30)
        correlacoes_nota = correlacoes['nota'].drop('nota').sort_values(ascending=False)

        # Passo 23: Classificar as correlações
        def classificar_correlacao(valor):
            if pd.isna(valor):
                return 'sem dados'
            if abs(valor) >= 0.7:
                return 'forte'
            elif abs(valor) >= 0.3:
                return 'média'
            else:
                return 'fraca'

        correlacoes_classificadas = correlacoes_nota.apply(classificar_correlacao)

        # Remover colunas com correlação absoluta >= 0.99
        extreme_correlation_cols = correlacoes_nota[correlacoes_nota.abs() >= 0.99].index.tolist()
        if extreme_correlation_cols:
            df_final = df_final.drop(columns=extreme_correlation_cols)
            valid_question_columns = [col for col in valid_question_columns if col not in extreme_correlation_cols]
    else:
        print("\nNenhuma coluna de pergunta válida foi selecionada para correlação.")

    # Passo 25: Função para criar o modelo binário e extrair top variáveis
    def criar_modelo_binario(df, classe_alvo, contexto, modelos_dict):
        df_bin = df.copy()
        df_bin['target_bin'] = np.where(df_bin['target'] == classe_alvo, 1, 0)

        # Selecionar X e y
        X = df_bin[valid_question_columns]
        y = df_bin['target_bin']

        # Remover colunas com alta proporção de NaNs
        high_nan_cols = X.columns[X.isnull().mean() > max_nan_ratio]
        if len(high_nan_cols) > 0:
            X = X.drop(columns=high_nan_cols)

        # Verificar se X e y possuem dados suficientes
        min_samples = 10  # Número mínimo de amostras
        if X.empty or y.empty or len(X) < min_samples:
            return None

        # Verificar se cada classe tem amostras suficientes
        class_counts = y.value_counts()
        if class_counts.min() < 3:
            return None

        # Divisão em treino e teste com estratificação
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError as e:
            return None

        # Modelo XGBoost
        modelo = XGBClassifier(eval_metric='logloss')
        modelo.fit(X_train, y_train)

        # Obter as top 10 variáveis com importâncias não nulas
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_vars = [X.columns[i] for i in indices if importances[i] > 0][:10]
        top_importances = [importances[i] for i in indices if importances[i] > 0][:10]

        if not top_vars:
            return None

        importances_percent = (np.array(top_importances) / np.sum(top_importances)) * 100

        # Armazenar o modelo e as top variáveis no dicionário modelos_dict
        modelos_dict[(contexto, classe_alvo)] = {
            'modelo': modelo,
            'top_vars': list(zip(top_vars, importances_percent))
        }

        return modelo

    # Passo 26: Dicionário para armazenar os modelos
    modelos_dict = {}

    # Passo 27: Modelos para a base total
    if not df_final.empty:
        criar_modelo_binario(df_final, 'detrator', 'Base Total', modelos_dict)
        criar_modelo_binario(df_final, 'neutro', 'Base Total', modelos_dict)
    else:
        print("\nErro: O DataFrame final está vazio. Nenhum modelo será treinado para a base total.")

    # Passo 28: Modelos por região
    if 'regiao' in df_final.columns:
        unique_regioes = sorted(df_final['regiao'].dropna().unique())
        for regiao in unique_regioes:
            df_regiao = df_final[df_final['regiao'] == regiao]
            contexto = f'Região {regiao}'
            criar_modelo_binario(df_regiao, 'detrator', contexto, modelos_dict)
            criar_modelo_binario(df_regiao, 'neutro', contexto, modelos_dict)
    else:
        print("\nAviso: A coluna 'regiao' não existe no DataFrame.")

    # Passo 29A: Modelos por período de pesquisa, se existir
    if 'periodo_de_pesquisa' in df_final.columns:
        unique_periodos = sorted(df_final['periodo_de_pesquisa'].dropna().unique())
        for periodo in unique_periodos:
            df_periodo = df_final[df_final['periodo_de_pesquisa'] == periodo]
            contexto = f'Período {periodo}'
            criar_modelo_binario(df_periodo, 'detrator', contexto, modelos_dict)
            criar_modelo_binario(df_periodo, 'neutro', contexto, modelos_dict)
    else:
        print("\nA coluna 'periodo_de_pesquisa' não existe no DataFrame.")

    # Passo 29B: Modelos por combinações de Região e Período
    if 'regiao' in df_final.columns and 'periodo_de_pesquisa' in df_final.columns:
        for regiao in unique_regioes:
            for periodo in unique_periodos:
                df_combination = df_final[(df_final['regiao'] == regiao) & (df_final['periodo_de_pesquisa'] == periodo)]
                if not df_combination.empty:
                    contexto = f'Região {regiao} - Período {periodo}'
                    criar_modelo_binario(df_combination, 'detrator', contexto, modelos_dict)
                    criar_modelo_binario(df_combination, 'neutro', contexto, modelos_dict)

    # Passo 30: Funções para salvar as tabelas de volumetria e correlação
    def gerar_tabelas_volumetria(df_final):
        os.makedirs('Resultados/Volumetria', exist_ok=True)
        caminho_excel = 'Resultados/Volumetria/Volumetria_Tabelas.xlsx'
        with pd.ExcelWriter(caminho_excel) as writer:
            volumetria_total = calcular_volumetria(df_final, ['safra'], "Volumetria por Safra - Base Total")
            volumetria_total.to_excel(writer, sheet_name='Base_Total_Safra')
            if 'regiao' in df_final.columns:
                for regiao in unique_regioes:
                    df_regiao = df_final[df_final['regiao'] == regiao]
                    volumetria_regiao = calcular_volumetria(df_regiao, ['safra'], f"Volumetria por Safra - Região {regiao}")
                    sheet_name = f"Regiao_{regiao}"[:31]
                    volumetria_regiao.to_excel(writer, sheet_name=sheet_name)
            if 'periodo_de_pesquisa' in df_final.columns:
                for periodo in unique_periodos:
                    df_periodo = df_final[df_final['periodo_de_pesquisa'] == periodo]
                    volumetria_periodo = calcular_volumetria(df_periodo, ['safra'], f"Volumetria por Safra - Período {periodo}")
                    sheet_name = f"Periodo_{periodo}"[:31]
                    volumetria_periodo.to_excel(writer, sheet_name=sheet_name)

    def gerar_tabelas_correlacao(df_final):
        os.makedirs('Resultados/Correlacao', exist_ok=True)
        caminho_excel = 'Resultados/Correlacao/Correlacao_Spearman.xlsx'
        with pd.ExcelWriter(caminho_excel) as writer:
            if 'nota' in df_final.columns and valid_question_columns:
                correlacoes = df_final[['nota'] + valid_question_columns].corr(method='spearman', min_periods=30)
                correlacoes_nota = correlacoes['nota'].drop('nota').sort_values(ascending=False)
                correlacoes_classificadas = correlacoes_nota.apply(classificar_correlacao)
                correlacoes_df = pd.DataFrame({
                    'Correlação': correlacoes_nota,
                    'Classificação': correlacoes_classificadas
                })
                correlacoes_df.to_excel(writer, sheet_name='Base_Total')
            if 'regiao' in df_final.columns:
                for regiao in unique_regioes:
                    df_regiao = df_final[df_final['regiao'] == regiao]
                    if not df_regiao.empty and 'nota' in df_regiao.columns and valid_question_columns:
                        correlacoes = df_regiao[['nota'] + valid_question_columns].corr(method='spearman', min_periods=30)
                        correlacoes_nota = correlacoes['nota'].drop('nota').sort_values(ascending=False)
                        correlacoes_classificadas = correlacoes_nota.apply(classificar_correlacao)
                        correlacoes_df = pd.DataFrame({
                            'Correlação': correlacoes_nota,
                            'Classificação': correlacoes_classificadas
                        })
                        sheet_name = f"Correlacao_Regiao_{regiao}"[:31]
                        correlacoes_df.to_excel(writer, sheet_name=sheet_name)
            if 'periodo_de_pesquisa' in df_final.columns:
                for periodo in unique_periodos:
                    df_periodo = df_final[df_final['periodo_de_pesquisa'] == periodo]
                    if not df_periodo.empty and 'nota' in df_periodo.columns and valid_question_columns:
                        correlacoes = df_periodo[['nota'] + valid_question_columns].corr(method='spearman', min_periods=30)
                        correlacoes_nota = correlacoes['nota'].drop('nota').sort_values(ascending=False)
                        correlacoes_classificadas = correlacoes_nota.apply(classificar_correlacao)
                        correlacoes_df = pd.DataFrame({
                            'Correlação': correlacoes_nota,
                            'Classificação': correlacoes_classificadas
                        })
                        sheet_name = f"Correlacao_Periodo_{periodo}"[:31]
                        correlacoes_df.to_excel(writer, sheet_name=sheet_name)
            if 'safra' in df_final.columns:
                for safra in df_final['safra'].dropna().unique():
                    df_safra = df_final[df_final['safra'] == safra]
                    if not df_safra.empty and 'nota' in df_safra.columns and valid_question_columns:
                        correlacoes = df_safra[['nota'] + valid_question_columns].corr(method='spearman', min_periods=30)
                        correlacoes_nota = correlacoes['nota'].drop('nota').sort_values(ascending=False)
                        correlacoes_classificadas = correlacoes_nota.apply(classificar_correlacao)
                        correlacoes_df = pd.DataFrame({
                            'Correlação': correlacoes_nota,
                            'Classificação': correlacoes_classificadas
                        })
                        sheet_name = f"Correlacao_Safra_{int(safra)}"[:31]
                        correlacoes_df.to_excel(writer, sheet_name=sheet_name)

    def salvar_top_variaveis(modelos_dict):
        os.makedirs('Resultados/Top_Variaveis', exist_ok=True)
        caminho_excel = 'Resultados/Top_Variaveis/Top_Variaveis_Modelos.xlsx'
        with pd.ExcelWriter(caminho_excel) as writer:
            for chave, info in modelos_dict.items():
                if isinstance(info, dict) and 'top_vars' in info:
                    contexto, classe_alvo = chave
                    top_vars = info['top_vars']
                    df_top_vars = pd.DataFrame(top_vars, columns=['Variavel', 'Importancia (%)'])
                    titulo = f"Top_Variaveis_{contexto}_{classe_alvo}"[:31]
                    df_top_vars.to_excel(writer, sheet_name=titulo, index=False)

    # Função para calcular a volumetria
    def calcular_volumetria(df, group_by_cols, titulo):
        volumetria = df.groupby(group_by_cols + ['target']).size().unstack(fill_value=0)
        volumetria['Total'] = volumetria.sum(axis=1)
        for categoria in categorias_target:
            if categoria not in volumetria.columns:
                volumetria[categoria] = 0
        volumetria['%Promotores'] = (volumetria['promotor'] / volumetria['Total']) * 100
        volumetria['%Neutros'] = (volumetria['neutro'] / volumetria['Total']) * 100
        volumetria['%Detratores'] = (volumetria['detrator'] / volumetria['Total']) * 100
        return volumetria

    # Função para classificar correlações
    def classificar_correlacao(valor):
        if pd.isna(valor):
            return 'sem dados'
        if abs(valor) >= 0.7:
            return 'forte'
        elif abs(valor) >= 0.3:
            return 'média'
        else:
            return 'fraca'

    # Passo 30: Salvar as tabelas de volumetria e correlação
    gerar_tabelas_volumetria(df_final)
    gerar_tabelas_correlacao(df_final)

    # Passo 31: Salvar as top 10 variáveis de cada modelo
    salvar_top_variaveis(modelos_dict)

    # Passo 32: Retornar as variáveis necessárias para a interface gráfica
    combined_contexts = [f'Região {regiao} - Período {periodo}' for regiao in unique_regioes for periodo in unique_periodos]

    return {
        'df': df_final,
        'modelos': modelos_dict,
        'contextos': ['Base Total'] + [f'Região {regiao}' for regiao in unique_regioes] + [f'Período {periodo}' for periodo in unique_periodos] + combined_contexts,
        'classes': ['detrator', 'neutro'],
        'tipos_grafico': ['Importância do Modelo', 'Valores SHAP'],
        'colunas_relevantes': valid_question_columns,
        'limpar_nomes_colunas': limpar_nomes_colunas
    }