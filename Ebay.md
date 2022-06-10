**Data Preparation & Machine Learning with Ebay Shill Bidding data**
================
Lucas Pontes
10 June 2022

-   [**Introduction**](#introduction)
-   [**Data preparation**](#data-preparation)
    -   [**Data characterisation**](#data-characterisation)
    -   [**Exploratory Graphs**](#exploratory-graphs)
-   [**Data cleaning**](#data-cleaning)
-   [**Feature engineering**](#feature-engineering)
-   [**Data scaling**](#data-scaling)
-   [**Linear Discriminant Analysis
    (LDA)**](#linear-discriminant-analysis-lda)

# **Introduction**

The Shill Bidding Data set `SBD` comprise eBay auctions with many
different features, including the duration of the auctions, bidder
tendency and class, among others (Alzahrani and Sadaoui, 2018). The aim
of this report is to submit the data to different supervised and
unsupervised machine learning methods after properly characterisation
and preparation of the dataset. To achieve the best result a scaling
method will be applied and compared alongside feature reduction methods,
machine-learning methods applied will also have the performance and
accuracy level measured and compared. At the end of this report, the
supervised and unsupervised methods will be identified in order to work
with optimal performance in the Shill Bidding Dataset. The predictions
related to normal and abnormal bidding behaviour of eBay users may help
companies to identify scams and other undesirable users within the
platform.

# **Data preparation**

## **Data characterisation**

Data characterisation is defined as the pre-processing phase of
summarization of the different features and characteristics presents in
the observations of a given data set, this process may be accomplished
by introducing the data to the viewer using statistics measure
summaries, and presenting visually using graphs, like bar charts and
scatter plots (Capozzoli, Cerquitelli and Piscitelli, 2016; Han, Kamber
and Pei, 2012).

``` r
SBD
```

    ## # A tibble: 6,321 × 13
    ##    Record_ID Auction_ID Bidder_ID Bidder_Tendency Bidding_Ratio Successive_Outb…
    ##        <dbl>      <dbl> <chr>               <dbl>         <dbl>            <dbl>
    ##  1         1        732 _***i              0.2            0.4                0  
    ##  2         2        732 g***r              0.0244         0.2                0  
    ##  3         3        732 t***p              0.143          0.2                0  
    ##  4         4        732 7***n              0.1            0.2                0  
    ##  5         5        900 z***z              0.0513         0.222              0  
    ##  6         8        900 i***e              0.0385         0.111              0  
    ##  7        10        900 m***p              0.4            0.222              0  
    ##  8        12        900 k***a              0.138          0.444              1  
    ##  9        13       2370 g***r              0.122          0.185              1  
    ## 10        27        600 e***t              0.155          0.346              0.5
    ## # … with 6,311 more rows, and 7 more variables: Last_Bidding <dbl>,
    ## #   Auction_Bids <dbl>, Starting_Price_Average <dbl>, Early_Bidding <dbl>,
    ## #   Winning_Ratio <dbl>, Auction_Duration <dbl>, Class <dbl>

``` r
glimpse(SBD)
```

    ## Rows: 6,321
    ## Columns: 13
    ## $ Record_ID              <dbl> 1, 2, 3, 4, 5, 8, 10, 12, 13, 27, 37, 38, 39, 4…
    ## $ Auction_ID             <dbl> 732, 732, 732, 732, 900, 900, 900, 900, 2370, 6…
    ## $ Bidder_ID              <chr> "_***i", "g***r", "t***p", "7***n", "z***z", "i…
    ## $ Bidder_Tendency        <dbl> 0.200000000, 0.024390244, 0.142857143, 0.100000…
    ## $ Bidding_Ratio          <dbl> 0.40000000, 0.20000000, 0.20000000, 0.20000000,…
    ## $ Successive_Outbidding  <dbl> 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.…
    ## $ Last_Bidding           <dbl> 0.0000277778, 0.0131226852, 0.0030416667, 0.097…
    ## $ Auction_Bids           <dbl> 0.00000000, 0.00000000, 0.00000000, 0.00000000,…
    ## $ Starting_Price_Average <dbl> 0.9935928, 0.9935928, 0.9935928, 0.9935928, 0.0…
    ## $ Early_Bidding          <dbl> 0.0000277778, 0.0131226852, 0.0030416667, 0.097…
    ## $ Winning_Ratio          <dbl> 0.6666667, 0.9444444, 1.0000000, 1.0000000, 0.5…
    ## $ Auction_Duration       <dbl> 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,…
    ## $ Class                  <dbl> 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0,…

``` r
summary(SBD)
```

    ##    Record_ID       Auction_ID    Bidder_ID         Bidder_Tendency  
    ##  Min.   :    1   Min.   :   5   Length:6321        Min.   :0.00000  
    ##  1st Qu.: 3778   1st Qu.: 589   Class :character   1st Qu.:0.02703  
    ##  Median : 7591   Median :1246   Mode  :character   Median :0.06250  
    ##  Mean   : 7536   Mean   :1241                      Mean   :0.14254  
    ##  3rd Qu.:11277   3rd Qu.:1867                      3rd Qu.:0.16667  
    ##  Max.   :15144   Max.   :2538                      Max.   :1.00000  
    ##  Bidding_Ratio     Successive_Outbidding  Last_Bidding      Auction_Bids   
    ##  Min.   :0.01176   Min.   :0.0000        Min.   :0.00000   Min.   :0.0000  
    ##  1st Qu.:0.04348   1st Qu.:0.0000        1st Qu.:0.04793   1st Qu.:0.0000  
    ##  Median :0.08333   Median :0.0000        Median :0.44094   Median :0.1429  
    ##  Mean   :0.12767   Mean   :0.1038        Mean   :0.46312   Mean   :0.2316  
    ##  3rd Qu.:0.16667   3rd Qu.:0.0000        3rd Qu.:0.86036   3rd Qu.:0.4545  
    ##  Max.   :1.00000   Max.   :1.0000        Max.   :0.99990   Max.   :0.7882  
    ##  Starting_Price_Average Early_Bidding     Winning_Ratio    Auction_Duration
    ##  Min.   :0.0000         Min.   :0.00000   Min.   :0.0000   Min.   : 1.000  
    ##  1st Qu.:0.0000         1st Qu.:0.02662   1st Qu.:0.0000   1st Qu.: 3.000  
    ##  Median :0.0000         Median :0.36010   Median :0.0000   Median : 5.000  
    ##  Mean   :0.4728         Mean   :0.43068   Mean   :0.3677   Mean   : 4.615  
    ##  3rd Qu.:0.9936         3rd Qu.:0.82676   3rd Qu.:0.8519   3rd Qu.: 7.000  
    ##  Max.   :0.9999         Max.   :0.99990   Max.   :1.0000   Max.   :10.000  
    ##      Class       
    ##  Min.   :0.0000  
    ##  1st Qu.:0.0000  
    ##  Median :0.0000  
    ##  Mean   :0.1068  
    ##  3rd Qu.:0.0000  
    ##  Max.   :1.0000

A first glance at `SBD` shows 6321 observations for 13 features, the
first three columns represent the record ID, auction and the bidder
respectively. The dplyr glimpse function displays all columns beside
bidder ID as numeric, however, class and all IDs columns should be
treated as a character class.

``` r
summary(SBD)
```

    ##   Record_ID          Auction_ID         Bidder_ID         Bidder_Tendency  
    ##  Length:6321        Length:6321        Length:6321        Min.   :0.00000  
    ##  Class :character   Class :character   Class :character   1st Qu.:0.02703  
    ##  Mode  :character   Mode  :character   Mode  :character   Median :0.06250  
    ##                                                           Mean   :0.14254  
    ##                                                           3rd Qu.:0.16667  
    ##                                                           Max.   :1.00000  
    ##  Bidding_Ratio     Successive_Outbidding  Last_Bidding      Auction_Bids   
    ##  Min.   :0.01176   Min.   :0.0000        Min.   :0.00000   Min.   :0.0000  
    ##  1st Qu.:0.04348   1st Qu.:0.0000        1st Qu.:0.04793   1st Qu.:0.0000  
    ##  Median :0.08333   Median :0.0000        Median :0.44094   Median :0.1429  
    ##  Mean   :0.12767   Mean   :0.1038        Mean   :0.46312   Mean   :0.2316  
    ##  3rd Qu.:0.16667   3rd Qu.:0.0000        3rd Qu.:0.86036   3rd Qu.:0.4545  
    ##  Max.   :1.00000   Max.   :1.0000        Max.   :0.99990   Max.   :0.7882  
    ##  Starting_Price_Average Early_Bidding     Winning_Ratio    Auction_Duration
    ##  Min.   :0.0000         Min.   :0.00000   Min.   :0.0000   Min.   : 1.000  
    ##  1st Qu.:0.0000         1st Qu.:0.02662   1st Qu.:0.0000   1st Qu.: 3.000  
    ##  Median :0.0000         Median :0.36010   Median :0.0000   Median : 5.000  
    ##  Mean   :0.4728         Mean   :0.43068   Mean   :0.3677   Mean   : 4.615  
    ##  3rd Qu.:0.9936         3rd Qu.:0.82676   3rd Qu.:0.8519   3rd Qu.: 7.000  
    ##  Max.   :0.9999         Max.   :0.99990   Max.   :1.0000   Max.   :10.000  
    ##  Class   
    ##  0:5646  
    ##  1: 675  
    ##          
    ##          
    ##          
    ## 

The summary function exhibits some general descriptive statistics, it is
noticeable that the data have been pre-processed as auction duration
range from 0 to 10 and all other numerical features range from 0 to 1.

## **Exploratory Graphs**

``` r
vis_miss(SBD)
```

    ## Warning: `gather_()` was deprecated in tidyr 1.2.0.
    ## Please use `gather()` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was generated.

![](Ebay_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

As displayed by the graphic generated by naniar function “vis_miss” on
`SBD`, the data does not include any standard missing value.

``` r
ggplot(prop, 
       aes(fill = Class, 
           area = n,
           label = c("Normal", "Abnormal"))) +
  geom_treemap() + 
  geom_treemap_text(colour = "white", 
                    place = "centre") +
  labs(title = "Proportion of normal and abnormal bidding behaviour")
```

![](Ebay_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
table(SBD$Class)
```

    ## 
    ##    0    1 
    ## 5646  675

``` r
round(prop.table(table(SBD$Class)),digits=2)
```

    ## 
    ##    0    1 
    ## 0.89 0.11

As displayed by the treemap and proportion table above, only 10.7% (675
observations) of the biddings on `SBD` are considered abnormal. The
graph above does a great job showing the proportion. The conventional
use of pie charts is not recommended to display proportion, especially
when the feature contains multiple unique observations or small
fractions, Human perception does not comprehend angular proportions in
pie charts as it should (Hunt, 2019).

``` r
ggplot(data = melted_corr_mat, aes(x=Var1, y=Var2,
                                   fill=value)) +
geom_tile() +
theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
scale_fill_viridis(discrete=FALSE) +
geom_text(aes(Var2, Var1, label = value),
          color = "black", size = 4) +
  labs(title = "Correlation Analysis") +
xlab("") +
ylab("")
```

![](Ebay_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The heatmap above shows the correlation between all numerical features
in a single graph. The problem with heatmaps is that slight colour
intensity variations can be hard to perceive by human eyes, especially
with certain colour combinations. Approximately 8% of men and 0.5% of
women have some form of colourblindness (BioTuring’s Blog, 2018; Colour
Blind Awareness, n.d.). The ggplot graph above has a colourblind
friendly colour palette and an added numerical scale, allowing the
viewer to check the exact correlation between the variables, as an
example, the positive correlation between early and last bidding
features is notorious. Another noticeable correlation is the negative
one between the winning ratio and auction bids, for further
investigation, both will be displayed in a scatter plot.

``` r
ggplot(SBD, aes(x = Early_Bidding, y=Last_Bidding )) + 
    geom_point(size=2,alpha= 0.3) +
    theme_ipsum() +
    ggtitle("Correlation between Early and Last Bidding")
```

![](Ebay_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

The scatter plot makes it noticeable why those two variables are so
positive correlated, considering that the last bidding value can never
be lower than the early bidding value. A positive slope can be found in
the diagonal centre of the 2-dimensional plot, this line represents the
values which were the same in both features.

``` r
ggplot(SBD, aes(x = Winning_Ratio, y=Auction_Bids )) + 
    geom_point(size=2,alpha= 0.3) +
    theme_ipsum()+
    ggtitle("Correlation between winning ratio and auction bids")
```

![](Ebay_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

Different from the previous scatter plot, this time the features auction
bids and winning ratio, which had a negative correlation in the heatmap
representation, do not show any visible correlation. This interaction
will be later tested when used as a predictor in a linear regression
method.

``` r
ggplot(SBD, aes(x= as.numeric(Record_ID))) +
  geom_histogram( binwidth=300, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Record ID distribution") +
  theme_ipsum() +
  theme(plot.title = element_text(size=15))
```

![](Ebay_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
ggplot(SBD, aes(x= as.numeric(Auction_ID))) +
  geom_histogram( binwidth=60, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Record ID distribution") +
  theme_ipsum() +
  theme(plot.title = element_text(size=15))
```

![](Ebay_files/figure-gfm/unnamed-chunk-12-2.png)<!-- -->

The histograms above display a clear difference between the distribution
of the Auction and Record ID, as the x-axis of the Auction ID, is much
shorter, this happens because many auctions are represented in the data
set multiple times. Also, the Record ID displays some uniformity, even
so not perfectly uniform, representing the absence of multiple
observations with the same ID.

``` r
ggplot(melt_SBD, aes(x=variable, y=value, fill=variable )) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha=0.6) +
  theme_ipsum() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ggtitle("Box plot for multiple variables") +
  xlab("")
```

![](Ebay_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

The Boxplot above shows the distribution of several numerical features
with a range between 0 and 1. These 8 features were chosen to be
represented by grouped boxplots since it is an exceptional alternative
to viewing many distributions in a single graph, especially if the
y-axis range matches all features. This figure is extremely informative,
considering that it displays the distribution of the observations, the
quartile range including the median, and the minimum and maximum values,
which describe the range (Galarnyk, 2018).

``` r
ggplot(SBD, aes(x= Auction_Duration)) +
  geom_histogram( binwidth=1, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Auction Duration distribution") +
  theme_ipsum() +
  theme(plot.title = element_text(size=15))
```

![](Ebay_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

Despite being a numerical feature, the low number of distinct values
causes the histogram to not be aesthetically pleasing, however, it shows
the 5 distinct values 7,3,1,5 and 10 and how many times each one
appeared in the auction duration column. This pattern suggests five
fixed auction durations options that the platform allows the sellers to
choose from.

``` r
SBD[,4:13] %>%
pivot_wider(names_from = Class,
            values_from = c("Bidder_Tendency", "Bidding_Ratio", "Successive_Outbidding", "Last_Bidding","Auction_Bids", 
                            "Starting_Price_Average" , "Early_Bidding" ,
                            "Winning_Ratio" , "Auction_Duration" ), values_fn = ~mean(.x)) %>%
as.data.frame() %>% 
t() %>%
kable(format = "html")
```

<table>
<tbody>
<tr>
<td style="text-align:left;">
Bidder_Tendency_0
</td>
<td style="text-align:right;">
0.1224033
</td>
</tr>
<tr>
<td style="text-align:left;">
Bidder_Tendency_1
</td>
<td style="text-align:right;">
0.3109790
</td>
</tr>
<tr>
<td style="text-align:left;">
Bidding_Ratio_0
</td>
<td style="text-align:right;">
0.1017746
</td>
</tr>
<tr>
<td style="text-align:left;">
Bidding_Ratio_1
</td>
<td style="text-align:right;">
0.3442676
</td>
</tr>
<tr>
<td style="text-align:left;">
Successive_Outbidding_0
</td>
<td style="text-align:right;">
0.0166490
</td>
</tr>
<tr>
<td style="text-align:left;">
Successive_Outbidding_1
</td>
<td style="text-align:right;">
0.8325926
</td>
</tr>
<tr>
<td style="text-align:left;">
Last_Bidding_0
</td>
<td style="text-align:right;">
0.4502861
</td>
</tr>
<tr>
<td style="text-align:left;">
Last_Bidding_1
</td>
<td style="text-align:right;">
0.5704626
</td>
</tr>
<tr>
<td style="text-align:left;">
Auction_Bids_0
</td>
<td style="text-align:right;">
0.2276376
</td>
</tr>
<tr>
<td style="text-align:left;">
Auction_Bids_1
</td>
<td style="text-align:right;">
0.2647969
</td>
</tr>
<tr>
<td style="text-align:left;">
Starting_Price_Average_0
</td>
<td style="text-align:right;">
0.4656051
</td>
</tr>
<tr>
<td style="text-align:left;">
Starting_Price_Average_1
</td>
<td style="text-align:right;">
0.5331813
</td>
</tr>
<tr>
<td style="text-align:left;">
Early_Bidding_0
</td>
<td style="text-align:right;">
0.4236299
</td>
</tr>
<tr>
<td style="text-align:left;">
Early_Bidding_1
</td>
<td style="text-align:right;">
0.4896741
</td>
</tr>
<tr>
<td style="text-align:left;">
Winning_Ratio_0
</td>
<td style="text-align:right;">
0.3082423
</td>
</tr>
<tr>
<td style="text-align:left;">
Winning_Ratio_1
</td>
<td style="text-align:right;">
0.8653224
</td>
</tr>
<tr>
<td style="text-align:left;">
Auction_Duration_0
</td>
<td style="text-align:right;">
4.5970599
</td>
</tr>
<tr>
<td style="text-align:left;">
Auction_Duration_1
</td>
<td style="text-align:right;">
4.7659259
</td>
</tr>
</tbody>
</table>

The pivot table above points to a substantial difference in the mean
value for most features when compared to normal and abnormal bidding
patterns. This information is important since the class feature will be
the dependable variable for machine learning models in the next steps.

# **Data cleaning**

The importance of identifying and removing missing values from the data
prior to modelling is that it can reduce its accuracy and cause bias in
the analysis (Analytics Vidhya, 2021). As shown before, there is not a
single missing value in the data. However, sometimes the missing values
appear as non-standard, a different format that includes strings named
“NA”, “n/a”, “–” and many other possibilities. Unfortunately, it is
near-impossible to check for all of them, especially in datasets with
numerous observations and features. When it happens the feature type can
only be categorical, since 0 in numeric features should not be
considered meaningless (Sullivan, 2018). This section aims to check and
remove non-standard missing values that may be present in the
“Bidder_ID” column.

``` r
SBD <- SBD %>%
  mutate(Bidder_ID = replace(Bidder_ID, Bidder_ID == "na", NA)) %>%
  mutate(Bidder_ID = replace(Bidder_ID, Bidder_ID == "N/A", NA)) %>%
  mutate(Bidder_ID = replace(Bidder_ID, Bidder_ID == "--", NA)) %>%
  mutate(Bidder_ID = replace(Bidder_ID, Bidder_ID == "n/a", NA)) 

sum(is.na(SBD$Bidder_ID))
```

    ## [1] 0

Checking for common strings associated with non-standard missing values,
the result of the analysis indicates the complete absence of them.

# **Feature engineering**

As an important stage preceding the machine learning modelling, this
task requires the judgement of the data analyst and numerous trials and
errors. Feature engineering is the act of transforming a given feature
space with different tools, especially mathematical, with the main goal
of improving the model performance (Khurana, Samulowitz and Turaga,
2018).The first step will be removing unnecessary columns for the future
tasks, the first three columns only include the IDs for the record,
auction and bidder respectively and do not add any significance to the
analysis.

``` r
SBD <- as.data.frame(SBD[,4:13])
head(SBD)
```

    ##   Bidder_Tendency Bidding_Ratio Successive_Outbidding Last_Bidding Auction_Bids
    ## 1      0.20000000     0.4000000                     0 0.0000277778            0
    ## 2      0.02439024     0.2000000                     0 0.0131226852            0
    ## 3      0.14285714     0.2000000                     0 0.0030416667            0
    ## 4      0.10000000     0.2000000                     0 0.0974768519            0
    ## 5      0.05128205     0.2222222                     0 0.0013177910            0
    ## 6      0.03846154     0.1111111                     0 0.0168435847            0
    ##   Starting_Price_Average Early_Bidding Winning_Ratio Auction_Duration Class
    ## 1              0.9935928  0.0000277778     0.6666667                5     0
    ## 2              0.9935928  0.0131226852     0.9444444                5     0
    ## 3              0.9935928  0.0030416667     1.0000000                5     0
    ## 4              0.9935928  0.0974768519     1.0000000                5     0
    ## 5              0.0000000  0.0012417328     0.5000000                7     0
    ## 6              0.0000000  0.0168435847     0.8000000                7     0

As observed by the correlation of early and last biding in the scatter
plot, the early bidding can never exceed the last bidding in value,
however, what eventually happens is that the auction has no increase in
price and culminate in the same value. To join the information of the
two columns in one and add extra information to the future predictive
model, a new column named “Bidding difference” will substitute them with
the existing incremented value calculated by the difference between
their prices.

``` r
SBD$Bidding_difference <- SBD$Last_Bidding - SBD$Early_Bidding
SBD <- as.data.frame(SBD[, c(-4,-7) ])
SBD <- SBD[order(SBD$Bidding_difference),]
head(SBD,30)
```

    ##      Bidder_Tendency Bidding_Ratio Successive_Outbidding Auction_Bids
    ## 193      0.142857143    0.07142857                     0   0.00000000
    ## 423      0.500000000    0.10000000                     0   0.00000000
    ## 792      0.015384615    0.02941176                     0   0.47058824
    ## 906      0.008403361    0.05263158                     0   0.05263158
    ## 931      0.040000000    0.11111111                     0   0.00000000
    ## 1036     0.125000000    0.07142857                     0   0.00000000
    ## 1147     0.031250000    0.01960784                     0   0.64705882
    ## 2028     0.062500000    0.04545454                     0   0.18181818
    ## 2042     0.100000000    0.04761905                     0   0.14285714
    ## 2165     0.008196721    0.04000000                     0   0.28000000
    ## 2166     0.025000000    0.04000000                     0   0.28000000
    ## 2170     0.200000000    0.02857143                     0   0.48571429
    ## 2703     0.014084507    0.03225807                     0   0.41935484
    ## 2923     0.022222222    0.07142857                     0   0.00000000
    ## 3096     0.025000000    0.07692308                     0   0.00000000
    ## 3638     0.020408163    0.05555556                     0   0.00000000
    ## 3930     0.000000000    0.04347826                     0   0.21739130
    ## 4879     0.100000000    0.06250000                     0   0.00000000
    ## 4917     0.022727273    0.07142857                     0   0.00000000
    ## 5022     0.166666667    0.16666667                     0   0.00000000
    ## 5129     0.055555556    0.05000000                     0   0.10000000
    ## 5916     0.500000000    0.02325581                     0   0.58139535
    ## 504      0.111111111    0.11111111                     0   0.00000000
    ## 2450     0.018518519    0.03225807                     0   0.41935484
    ## 5666     0.200000000    0.02000000                     0   0.64000000
    ## 1        0.200000000    0.40000000                     0   0.00000000
    ## 2        0.024390244    0.20000000                     0   0.00000000
    ## 3        0.142857143    0.20000000                     0   0.00000000
    ## 4        0.100000000    0.20000000                     0   0.00000000
    ## 6        0.038461538    0.11111111                     0   0.00000000
    ##      Starting_Price_Average Winning_Ratio Auction_Duration Class
    ## 193               0.0000000     0.0000000                3     0
    ## 423               0.9935928     1.0000000                3     0
    ## 792               0.9935928     0.0000000                7     0
    ## 906               0.0000000     0.0000000                7     0
    ## 931               0.6764695     1.0000000                7     0
    ## 1036              0.9935928     0.0000000                3     0
    ## 1147              0.9935928     0.0000000                3     0
    ## 2028              0.9935928     0.0000000                3     0
    ## 2042              0.0000000     0.0000000                3     0
    ## 2165              0.9935928     0.0000000                7     0
    ## 2166              0.9935928     0.0000000                7     0
    ## 2170              0.9935928     0.0000000                3     0
    ## 2703              0.0000000     0.0000000                7     0
    ## 2923              0.0000000     0.0000000                7     0
    ## 3096              0.0000000     0.0000000                3     0
    ## 3638              0.0000000     0.0000000                7     0
    ## 3930              0.9935928     0.0000000                3     0
    ## 4879              0.0000000     0.0000000                7     0
    ## 4917              0.0000000     0.0000000                3     0
    ## 5022              0.0000000     0.6666667                3     0
    ## 5129              0.6764695     0.0000000                3     0
    ## 5916              0.9935281     0.0000000                3     0
    ## 504               0.0000000     0.8333333                7     0
    ## 2450              0.9935928     0.0000000                7     0
    ## 5666              0.0000000     0.0000000                3     0
    ## 1                 0.9935928     0.6666667                5     0
    ## 2                 0.9935928     0.9444444                5     0
    ## 3                 0.9935928     1.0000000                5     0
    ## 4                 0.9935928     1.0000000                5     0
    ## 6                 0.0000000     0.8000000                7     0
    ##      Bidding_difference
    ## 193        -1.00000e-10
    ## 423        -1.00000e-10
    ## 792        -1.00000e-10
    ## 906        -1.00000e-10
    ## 931        -1.00000e-10
    ## 1036       -1.00000e-10
    ## 1147       -1.00000e-10
    ## 2028       -1.00000e-10
    ## 2042       -1.00000e-10
    ## 2165       -1.00000e-10
    ## 2166       -1.00000e-10
    ## 2170       -1.00000e-10
    ## 2703       -1.00000e-10
    ## 2923       -1.00000e-10
    ## 3096       -1.00000e-10
    ## 3638       -1.00000e-10
    ## 3930       -1.00000e-10
    ## 4879       -1.00000e-10
    ## 4917       -1.00000e-10
    ## 5022       -1.00000e-10
    ## 5129       -1.00000e-10
    ## 5916       -1.00000e-10
    ## 504        -1.00000e-10
    ## 2450       -1.00000e-10
    ## 5666       -9.99999e-11
    ## 1           0.00000e+00
    ## 2           0.00000e+00
    ## 3           0.00000e+00
    ## 4           0.00000e+00
    ## 6           0.00000e+00

Calculating the difference between last and early bidding resulted in a
feature with 25 negative values unexpected, suggesting that in these 25
observations the trade was concluded with the last bidding having a
lower value than the early bidding, which is counterintuitive.

``` r
SBD$Auction_Duration <- SBD$Auction_Duration / 10

ggplot(SBD, aes(x= Auction_Duration)) +
  geom_histogram( binwidth=0.1, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Scaled Auction Duration distribution") +
  theme_ipsum() +
  theme(plot.title = element_text(size=15))
```

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

    ## Warning in grid.Call.graphics(C_text, as.graphicsAnnot(x$label), x$x, x$y, :
    ## font family not found in Windows font database

    ## Warning in grid.Call(C_textBounds, as.graphicsAnnot(x$label), x$x, x$y, : font
    ## family not found in Windows font database

![](Ebay_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

Since all other metrics range from 0 to 1, dividing the auction duration
variable by 10 puts it in the same range. This may reduce the chance of
this variable disproportionately impacting the model.

# **Data scaling**

The different orders of magnitude may bias the results since most
machine learning algorithms calculate the output using Euclidean
distance (Asaithambi, 2017). As seen in the distribution of the previous
boxplots and histograms the numeric features contain some outliers,
considering the putative relevance of these outliers in the machine
learning analysis, the ideal transformation method will maintain the
impact of these feature outliers in the output of the machine learning
algorithm to be performed. The two regularly used scaling methods are
standardization and normalization, for this dataset the standardization
will be applied as the transformation method of choice, since it copes
better with outliers than normalization methods (GeeksforGeeks, 2020).

As displayed before, except for a few negative values in the Bidding
difference column, all features range from 0 to 1. Data scaling is
necessary when the features of a dataset have a different range or order
of magnitude (Roy, 2020), which is not the case, taking that into
consideration the data scaling may not be necessary for that instance.
However, to evaluate that, the “X_s” variable will contain that scaled
data to be compared to the not-scaled data.

``` r
X <- SBD[,-8]
y <- SBD[,8]
X_s <- scale(X)
head(X)
```

    ##      Bidder_Tendency Bidding_Ratio Successive_Outbidding Auction_Bids
    ## 193      0.142857143    0.07142857                     0   0.00000000
    ## 423      0.500000000    0.10000000                     0   0.00000000
    ## 792      0.015384615    0.02941176                     0   0.47058824
    ## 906      0.008403361    0.05263158                     0   0.05263158
    ## 931      0.040000000    0.11111111                     0   0.00000000
    ## 1036     0.125000000    0.07142857                     0   0.00000000
    ##      Starting_Price_Average Winning_Ratio Auction_Duration Bidding_difference
    ## 193               0.0000000             0              0.3             -1e-10
    ## 423               0.9935928             1              0.3             -1e-10
    ## 792               0.9935928             0              0.7             -1e-10
    ## 906               0.0000000             0              0.7             -1e-10
    ## 931               0.6764695             1              0.7             -1e-10
    ## 1036              0.9935928             0              0.3             -1e-10

``` r
head(y)
```

    ## [1] 0 0 0 0 0 0
    ## Levels: 0 1

``` r
head(X_s)
```

    ##      Bidder_Tendency Bidding_Ratio Successive_Outbidding Auction_Bids
    ## 193       0.00160544    -0.4275901            -0.3710468   -0.9073611
    ## 423       1.81374474    -0.2103671            -0.3710468   -0.9073611
    ## 792      -0.64518890    -0.7470356            -0.3710468    0.9362609
    ## 906      -0.68061171    -0.5704999            -0.3710468   -0.7011665
    ## 931      -0.52029068    -0.1258915            -0.3710468   -0.9073611
    ## 1036     -0.08900153    -0.4275901            -0.3710468   -0.9073611
    ##      Starting_Price_Average Winning_Ratio Auction_Duration Bidding_difference
    ## 193              -0.9651145    -0.8423123       -0.6547772         -0.2698754
    ## 423               1.0629895     1.4482529       -0.6547772         -0.2698754
    ## 792               1.0629895    -0.8423123        0.9668691         -0.2698754
    ## 906              -0.9651145    -0.8423123        0.9668691         -0.2698754
    ## 931               0.4156830     1.4482529        0.9668691         -0.2698754
    ## 1036              1.0629895    -0.8423123       -0.6547772         -0.2698754

The scaled variables are going to be had their performance measured in
the feature reducing methods and the Random Forest Classification
machine learning method to come, after the comparison, the one with
better predictive power will be applied in the following tasks.

# **Linear Discriminant Analysis (LDA)**

In pattern classification using a Machine Learning algorithm, the
training vectors are evaluated by comparing them to the test split, a
common problem that can reduce its accuracy is the large dimensionality
of the data, and numerous features reducing the robustness of the
pattern classifiers. Linear discriminant analysis or LDA is a supervised
technique that generally exceeds other techniques when applied to
feature classification, LDA method reduces the number of features that
fit in the different classes prior to classification while maintaining
the classes well separated in this lower dimensional space (Sharma and
Paliwal, 2014).
