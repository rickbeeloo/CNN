library(ggplot2)
library(reshape2)

############
# GPU vs CPU 
############
gpu_cpu_file = read.table(file.choose(), sep = '\t', header = F)
colnames(gpu_cpu_file) <- c('epoch', 'GPU', 'CPU')
gpu_cpu_file$CPU= gpu_cpu_file$CPU

# convert to long format
performance.data <- melt(gpu_cpu_file, id="epoch")

# plotting
ggplot(data=performance.data,
       aes(x=epoch, y=value, colour=variable)) +
  geom_line(size = 2) +
  scale_color_manual(values=c("#CC6666", "#9999CC",'red')) +
  ggtitle('GPU vs CPU comparison') +
  xlab("#epochs") +
  ylab("Run time(sec)") +
  theme(plot.title = element_text(hjust = 0.5, size = 20,face="bold"),
        axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"))

# speed up
gpu_cpu_file$acceleration = gpu_cpu_file$CPU/gpu_cpu_file$GPU










