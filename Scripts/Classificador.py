from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas

def exibirResultado(valor):
    frase, resultado = valor
    resultado = 'Frase positiva' if resultado[0] == '1' else 'Frase negativa'
    print(frase, ':', resultado)


def analisarFrase(classificador, vetorizador, frase):
    return frase, classificador.predict(vetorizador.transform([frase]))

def obterDado():

    with open("imdb_labelled.txt", "r") as arquivo:
        dados = arquivo.read().split('\n')

    with open("amazon_cells_labelled.txt", "r") as arquivo:
        dados += arquivo.read().split('\n')

    with open("yelp_labelled.txt", "r") as arquivo:
        dados += arquivo.read().split('\n')

    return dados

def tratarDados(dados):
    dadosTratados = []

    for dado in dados:
        if len(dado.split("\t"))== 2 and dado.split("\t")[1] != "":
            dadosTratados.append(dado.split('\t'))
    return dadosTratados

def treinoValidacao(dados):
    quantTotal = len(dados)
    percentTreino = 0.75
    treino = []
    validacao = []

    for indice in range(0, quantTotal):
        if indice < quantTotal * percentTreino:
            treino.append(dados[indice])
        else:
            validacao.append((dados[indice]))

    return treino,validacao

def preProcessamento():
    dados = obterDado()
    dadosTratados = tratarDados(dados)

    return treinoValidacao(dadosTratados)

def realizarTreinamento(registrosTreino, vetorizador):
    treinoComentarios = [registros_Treino[0] for registros_Treino in registrosTreino]
    treinoRespostas = [registros_Treino[1] for registros_Treino in registrosTreino]

    treinoComentarios = vetorizador.fit_transform(treinoComentarios)

    return BernoulliNB().fit(treinoComentarios, treinoRespostas)

def realizarAvaliacaoSimples(registroAvaliacao):
    avaliacaoComentarios = [registro_Avaliacao[0] for registro_Avaliacao in registroAvaliacao]
    avaliacaoRespostas =  [registro_Avaliacao[1] for registro_Avaliacao in registroAvaliacao]

    total = len(avaliacaoComentarios)
    acertos = 0
    for indice in range(0, total):
        resultadoAnalise = analisarFrase((classificador, vetorizador, avaliacaoComentarios[indice]))
        frase, resultado = resultadoAnalise
        acertos += 1 if resultado[0] == avaliacaoRespostas else 0

        return acertos * 100 / total

def realizarAvaliacaoCompleta(registroAvaliacao):
    avaliacaoComentarios = [registro_Avaliacao[0] for registro_Avaliacao in registroAvaliacao]
    avaliacaoRespostas   = [registro_Avaliacao[1] for registro_Avaliacao in registroAvaliacao]

    total = len(avaliacaoComentarios)
    verdadeirosNegativo = 0
    verdadeirosPositivos = 0
    falsos_positivos = 0
    falsosNegativos = 0

    for indice in range(0, total):
        resultadoAnalise = analisarFrase(classificador, vetorizador, avaliacaoComentarios[indice])
        frase, resultado = resultadoAnalise
        if resultado[0] == '0':
            verdadeirosNegativo += 1 if avaliacaoRespostas[indice] == '0' else 0
            falsosNegativos += 1 if avaliacaoRespostas[indice] != '0' else 0
        else:
            verdadeirosPositivos += 1 if avaliacaoRespostas[indice] == '1' else 0
            falsos_positivos += 1 if avaliacaoRespostas[indice] != '1' else 0

    return ( verdadeirosPositivos * 100 / total,
             verdadeirosNegativo * 100 / total,
             falsos_positivos * 100 / total,
             falsosNegativos * 100 / total
           )

def matrizConfusao(registroAvaliacao):
    avaliacaoTexto = [registro_Avaliacao[0] for registro_Avaliacao in registroAvaliacao]
    resultadoAtual = [registro_Avaliacao[1] for registro_Avaliacao in registroAvaliacao]
    resultadoPredicao = []
    for text in avaliacaoTexto:
        analysis_result = analisarFrase(classificador, vetorizador, text)
        resultadoPredicao.append(analysis_result[1][0])

    matrix = confusion_matrix(resultadoAtual, resultadoPredicao)
    return matrix

#Chamadas de Função
registrosTreino, registroValidacao = preProcessamento()
vetorizador = CountVectorizer(binary = 'true')
classificador = realizarTreinamento(registrosTreino, vetorizador)
matrizResultado = matrizConfusao(registroValidacao)

#Print do analisador da frase
exibirResultado(analisarFrase(classificador, vetorizador, 'this is the best movie'))
exibirResultado(analisarFrase(classificador, vetorizador, 'this is the worst movie'))

#Criar DataFrame
df = pandas.DataFrame(matrizResultado, columns = ['Negativos', 'Positivos'], index = ['Negativos', 'Positivos'])
print(df)





classes = ["Negativos", "Positivos"]

plt.figure()
plt.imshow(matrizResultado, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz Confusão - Análise de Sentimentos")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

text_format = 'd'
thresh = matrizResultado.max() / 2.
for row, column in itertools.product(range(matrizResultado.shape[0]), range(matrizResultado.shape[1])):
    plt.text(column, row, format(matrizResultado[row, column], text_format),
             horizontalalignment="center",
             color="white" if matrizResultado[row, column] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

plt.show()



verdadeirosNegativo = matrizResultado[0][0]
falsos_negativos = matrizResultado[0][1]
falsos_positivos = matrizResultado[1][0]
verdadeirosPositivos = matrizResultado[1][1]

Acuracia = (verdadeirosPositivos + verdadeirosNegativo) / (verdadeirosPositivos + verdadeirosNegativo + falsos_positivos + falsos_negativos)
precisao = verdadeirosPositivos / (verdadeirosPositivos + falsos_negativos)
recall = verdadeirosPositivos / (verdadeirosPositivos + verdadeirosNegativo)
f1_score = 2*(recall * precisao) / (recall + precisao)

print('Acurácia:',Acuracia)
print('Precisão:',precisao)
print('Recall:',recall)
print('F1 pontuação:',f1_score)