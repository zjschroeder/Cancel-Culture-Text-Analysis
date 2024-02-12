# Utilizing PCA for Meaning Extraction Method
# Adapted from:
#-RYAN BOYD 
#-Lancaster University

library(psych)
library(GPArotation)
library(MASS)
here::here()

extractorRData <- function(file, object) {
  E <- new.env()
  load(file=file, envir=E)
  return(get(object, envir=E, inherits=F))
}

DF <- extractorRData("data/study5_dfm.RData", "tweets_binary_matrix")
DF[BeginningColumn:length(DF)] <- apply(DF[BeginningColumn:length(DF)], 2, as.character)
DF[BeginningColumn:length(DF)] <- apply(DF[BeginningColumn:length(DF)], 2, as.numeric)
colnames(DF)[1] = 'Filename'

# load("data")

BeginningColumn <- 1


kmo = function( data ){
  X <- cor(as.matrix(data)) 
  iX <- ginv(X) 
  S2 <- diag(diag((iX^-1)))
  AIS <- S2%*%iX%*%S2                      # anti-image covariance matrix
  IS <- X+AIS-2*S2                         # image covariance matrix
  Dai <- sqrt(diag(diag(AIS)))
  IR <- ginv(Dai)%*%IS%*%ginv(Dai)         # image correlation matrix
  AIR <- ginv(Dai)%*%AIS%*%ginv(Dai)       # anti-image correlation matrix
  a <- apply((AIR - diag(diag(AIR)))^2, 2, sum)
  AA <- sum(a) 
  b <- apply((X - diag(nrow(X)))^2, 2, sum)
  BB <- sum(b)
  MSA <- b/(b+a)                        # indiv. measures of sampling adequacy
  AIR <- AIR-diag(nrow(AIR))+diag(MSA)  # Examine the anti-image of the correlation matrix. That is the  negative of the partial correlations, partialling out all other variables.
  kmo <- BB/(AA+BB)                     # overall KMO statistic
  # Reporting the conclusion 
  if (kmo >= 0.00 && kmo < 0.50){test <- 'The KMO test yields a POOR degree of common variance.'} 
  else if (kmo >= 0.50 && kmo < 0.60){test <- 'The KMO test yields a SOMEWHAT POOR degree of common variance.'} 
  else if (kmo >= 0.60 && kmo < 0.70){test <- 'The KMO test yields a DECENT degree of common variance.'} 
  else if (kmo >= 0.70 && kmo < 0.80){test <- 'The KMO test yields a GOOD degree of common variance.' } 
  else if (kmo >= 0.80 && kmo < 0.90){test <- 'The KMO test yields a VERY GOOD degree of common variance.' }
  else { test <- 'The KMO test yields a FANTASTIC degree of common variance.' }
  
  ans <- list( overall = kmo,
               report = test,
               individual = MSA,
               AIS = AIS,
               AIR = AIR )
  return(ans)
} 

Bartlett.sphericity.test <- function(x)
{
  method <- "Bartlett's test of sphericity"
  data.name <- deparse(substitute(x))
  x <- subset(x, complete.cases(x))
  n <- nrow(x)
  p <- ncol(x)
  chisq <- (1-n+(2*p+5)/6)*log(det(cor(x)))
  df <- p*(p-1)/2
  p.value <- pchisq(chisq, df, lower.tail=FALSE)
  names(chisq) <- "X-squared"
  names(df) <- "df"
  return(structure(list(statistic=chisq, parameter=df, p.value=p.value, method=method, data.name=data.name), class="htest"))
}


remove_zero_var <- function(dat) {
  out <- lapply(dat, function(x) length(unique(x)))
  want <- which(out > 1)
  return(dat[, c(want)])
}


PCAFunction <- function(InputData, NumberOfFactors) {
  
  #remove columns with no variance
  InputData = remove_zero_var(InputData)
  
  #gets our scree plot (uncomment next line to get the plot)
  fa.parallel(InputData, fa="PC", main="Scree Plot")
  
  options(width=10000)
  options(max.print=2000000000)
  
  #runs the KMO test and prints our results
  cat("\n\nKMO_TEST:\n\n")
  KMO_Results <- kmo(InputData)
  cat(paste("KMO_METRIC: ", KMO_Results$overall, "\n", KMO_Results$report, sep=""))
  
  #same for the Bartlett test of sphericity
  cat("\n\nBARTLETT_SPHERICITY_TEST:\n\n")
  print(Bartlett.sphericity.test(InputData))
  
  #runs the PCA and prints the results
  cat("\n\nFACTOR_ANALYSIS_RESULTS:\n\n")
  PCA <- principal(InputData, nfactors=NumberOfFactors, residuals=FALSE, rotate="varimax", method="regression")
  
  print(PCA)
  
  return(PCA)
  
}




