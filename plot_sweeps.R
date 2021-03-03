setwd("D:/MCV_671DRDOG/psud/sweeps")
library(ggplot2)
library(dplyr)
library(ggpubr)
df <- read.csv("p25direct.csv")

df$Length <- as.factor(df$Length)
df$Threshold <- as.factor(df$Threshold)

#TODO: Can we call R from python

plots <- lapply(unique(df$Length),
       function(x){
         fdf <- filter(df, Length == x)
         ewc <- filter(fdf, Method == "EWC")
         g<- ggplot(fdf,aes(x=Weight,y=PSuD,color=Threshold,linetype=Method))+
           geom_line()+
           geom_point()
         for(thresh in ewc$Threshold){
           yval <- ewc[ewc$Threshold ==thresh,"PSuD"]
           g<- g +
             geom_hline(yintercept = yval,linetype="dashed")
         }
         ptitle <- paste("message length",x)
         mval <- 0

         g <- g+
           scale_y_continuous(breaks=seq(from=mval,to=1,by=0.2),
                              minor_breaks = seq(from=mval,to=1,by=0.1),
                              limits = c(mval,1))+
           ggtitle(ptitle)+
           theme_minimal()
       })

ggarrange(plotlist=plots,nrow=1,ncol=length(plots))
