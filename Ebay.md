**Data Preparation & Machine Learning with Ebay Shill Bidding data**
================
Lucas Pontes
01 June 2022

-   [**Introduction**](#introduction)
-   [**Data preparation**](#data-preparation)
    -   [**Data characterisation**](#data-characterisation)

# **Introduction**

The Shill Bidding Data set (`SBD`) comprise eBay auctions with many
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

### Exploratory Graphs

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
  labs(title = "Correlation Analysis")
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
