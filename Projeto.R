# Instala os pacotes necessários, se ainda não estiverem instalados
install.packages("OpenImageR")
install.packages("randomForest")

# Carrega as bibliotecas
library(OpenImageR)
library(randomForest)

# Carrega as imagens de treino
ceu_limpo_img = readImage("ceu_azul.png")
nuvens_brancas_img = readImage("ceu_branco.png")
nuvens_chuva_img = readImage("ceu_preto.png")

# Função auxiliar para extrair pixels e criar um data frame
extrair_pixels = function(imagem, rotulo_classe) {
  matriz_pixels = cbind(c(imagem[,,1]), c(imagem[,,2]), c(imagem[,,3]))
  df = as.data.frame(matriz_pixels)
  colnames(df) = c("R", "G", "B")
  df$Classe = rotulo_classe
  return(df)
}

# Rótulos: 1 = Céu Limpo, 2 = Nuvens Brancas, 3 = Nuvens de Chuva
df_ceu_limpo = extrair_pixels(ceu_limpo_img, 1)
df_nuvens_brancas = extrair_pixels(nuvens_brancas_img, 2)
df_nuvens_chuva = extrair_pixels(nuvens_chuva_img, 3)


# Reduz o tamanho da amostra para um treino mais rápido.
tamanho_amostra = 5000 
set.seed(42) 

#Amostragem:
amostra_ceu_limpo = df_ceu_limpo[sample(nrow(df_ceu_limpo), tamanho_amostra), ]
amostra_nuvens_brancas = df_nuvens_brancas[sample(nrow(df_nuvens_brancas), tamanho_amostra), ]
amostra_nuvens_chuva = df_nuvens_chuva[sample(nrow(df_nuvens_chuva), tamanho_amostra), ]

# Combina todas as amostras num único data frame
dados_treino = rbind(amostra_ceu_limpo, amostra_nuvens_brancas, amostra_nuvens_chuva)
dados_treino$Classe = as.factor(dados_treino$Classe)

### TREINO DO MODELO RANDOM FOREST
modelo_rf = randomForest(Classe ~ R + G + B, 
                         data = dados_treino, 
                         ntree = 200, 
                         importance = TRUE)

print(modelo_rf)
print(importance(modelo_rf))


###PREDIÇÃO NA IMAGEM ALVO
imagem_alvo = readImage("nuvem_total.png")

# Extrai TODOS os pixels da imagem alvo para a predição
df_alvo = extrair_pixels(imagem_alvo, NA)

cat("A realizar a predição em cada pixel da imagem alvo. Isto pode demorar...\n")
predicoes = predict(modelo_rf, newdata = df_alvo)
cat("Predição concluída.\n\n")


### ANÁLISE E VISUALIZAÇÃO DOS RESULTADOS
contagem_classes = table(predicoes)
percentagens = prop.table(contagem_classes) * 100

# Mapeia os rótulos numéricos para nomes descritivos
nomes_classes = c("1" = "Céu Limpo", "2" = "Nuvens Brancas", "3" = "Nuvens de Chuva")
names(percentagens) = nomes_classes[names(percentagens)]

print(round(percentagens, 2))


# Converte as predições (que são fatores) para valores numéricos
predicoes_numericas = as.numeric(as.character(predicoes))

# Cria uma imagem em branco com as mesmas dimensões da imagem alvo
imagem_segmentada_colorida = array(0, dim = dim(imagem_alvo))

# Define as cores para cada classe (valores entre 0 e 1 para R, G, B)
cor_ceu_limpo    = c(0.0, 0.4, 1.0) # Azul Brilhante
cor_nuvem_branca = c(1.0, 1.0, 1.0) # Branco
cor_nuvem_chuva  = c(0.2, 0.2, 0.2) # Cinzento Escuro

# Atribui a cor a cada pixel com base na sua classe predita, especificando o índice do canal de cor
# Canal Vermelho
imagem_segmentada_colorida[,,1] = ifelse(predicoes_numericas == 1, cor_ceu_limpo, ifelse(predicoes_numericas == 2, cor_nuvem_branca, cor_nuvem_chuva))
# Canal Verde
imagem_segmentada_colorida[,,2] = ifelse(predicoes_numericas == 1, cor_ceu_limpo, ifelse(predicoes_numericas == 2, cor_nuvem_branca, cor_nuvem_chuva))
# Canal Azul
imagem_segmentada_colorida[,,3] = ifelse(predicoes_numericas == 1, cor_ceu_limpo, ifelse(predicoes_numericas == 2, cor_nuvem_branca, cor_nuvem_chuva))


# Exibe as imagens
cat("A exibir os resultados visuais. Feche cada janela de imagem para ver a seguinte.\n")

# 1. Imagem Original
cat("1. Imagem Original\n")
imageShow(imagem_alvo)

# 2. Mapa de Segmentação Colorido
cat("2. Mapa de Segmentação (Azul=Céu, Branco=Nuvens Claras, Cinzento=Nuvens Escuras)\n")
imageShow(imagem_segmentada_colorida)

# --- Visualização Adicional: Destaque de cada classe na imagem original ---
cat("3. Destaque das áreas classificadas como 'Céu Limpo' (em azul)\n")
img_destaque_ceu = imagem_alvo
pixels_ceu_limpo = predicoes_numericas == 1
img_destaque_ceu[,,1][pixels_ceu_limpo] = cor_ceu_limpo # Usa apenas o componente R da cor
img_destaque_ceu[,,2][pixels_ceu_limpo] = cor_ceu_limpo # Usa apenas o componente G da cor
img_destaque_ceu[,,3][pixels_ceu_limpo] = cor_ceu_limpo # Usa apenas o componente B da cor
imageShow(img_destaque_ceu)

cat("4. Destaque das áreas classificadas como 'Nuvens Brancas' (em branco)\n")
img_destaque_brancas = imagem_alvo
pixels_nuvens_brancas = predicoes_numericas == 2
img_destaque_brancas[,,1][pixels_nuvens_brancas] = cor_nuvem_branca
img_destaque_brancas[,,2][pixels_nuvens_brancas] = cor_nuvem_branca
img_destaque_brancas[,,3][pixels_nuvens_brancas] = cor_nuvem_branca
imageShow(img_destaque_brancas)

cat("5. Destaque das áreas classificadas como 'Nuvens de Chuva' (em cinzento escuro)\n")
img_destaque_chuva = imagem_alvo
pixels_nuvens_chuva = predicoes_numericas == 3
img_destaque_chuva[,,1][pixels_nuvens_chuva] = cor_nuvem_chuva
img_destaque_chuva[,,2][pixels_nuvens_chuva] = cor_nuvem_chuva
img_destaque_chuva[,,3][pixels_nuvens_chuva] = cor_nuvem_chuva
imageShow(img_destaque_chuva)


