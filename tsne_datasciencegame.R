library(Rtsne)
library(data.table)
library(ggplot2)
library(readr)
library(Rtsne)

set.seed(42)
num_rows_sample <- 10000

train = fread('/Users/Hannes/Desktop/City University/datasciencegame/data2/df_train.csv', header=TRUE, data.table=F)
#train_sample <- train[sample(1:nrow(train), size=num_rows_sample), ]

#x = train_sample[, -17]

x_tmp = train[1813000:nrow(train), ]
x = x_tmp[, -17]
tsne <- Rtsne(as.matrix(x), dims=2, check_duplicates=FALSE,
              pca=TRUE, perplexity=30, theta=0.5, verbose=TRUE, max_iter=500)

# Visualisations

# Version 1 (best, with labels, through ggplot2)
embedding <- as.data.frame(tsne$Y)
embedding$Class <- as.factor(x_tmp$is_listened)

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  guides(colour=guide_legend(override.aes = list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE 2D Embedding") +
  theme_light(base_size=15) +  # size of title and legend
  theme(strip.background = element_blank(),
        strip.text.x     = element_blank(),
        axis.text.x      = element_blank(),
        axis.text.y      = element_blank(),
        axis.ticks       = element_blank(),
        axis.line        = element_blank(),
        panel.border     = element_blank())

ggsave("tsne_datasciencegame.pdf", p, width=8, height=6, units="in")

