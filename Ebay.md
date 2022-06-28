**Data Preparation & Machine Learning with Ebay Shill Bidding data**
================
Lucas Pontes
28 June 2022

-   [**Introduction**](#introduction)
-   [**Data preparation**](#data-preparation)
    -   [**Data characterisation**](#data-characterisation)
    -   [**Exploratory Data Analysis**](#exploratory-data-analysis)
    -   [**Data cleaning**](#data-cleaning)
    -   [**Feature engineering**](#feature-engineering)
    -   [**Data scaling**](#data-scaling)
-   [**Dimensionality reduction**](#dimensionality-reduction)
    -   [**Principal Component Analysis
        (PCA)**](#principal-component-analysis-pca)
    -   [**Linear Discriminant Analysis
        (LDA)**](#linear-discriminant-analysis-lda)
-   [**Machine Learning**](#machine-learning)
    -   [**Clustering algorithms**](#clustering-algorithms)
    -   [**Classification algorithms**](#classification-algorithms)
-   [**Conclusion**](#conclusion)
-   [**References**](#references)

**[SBD Dataset Web
Page](https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset)**

# **Introduction**

The Shill Bidding Data set `SBD` comprise eBay auctions with many
different features, including the duration of the auctions, bidder
tendency and class, among others (Alzahrani and Sadaoui, 2018). The aim
of this report is to submit the data to different supervised and
unsupervised machine learning methods after properly characterisation
and preparation of the dataset. To achieve the best result a scaling
method is applied alongside feature reduction methods, machine-learning
methods applied had their performance and accuracy level measured and
compared. At the end of this report, the supervised and unsupervised
methods were identified in order to work with optimal performance in the
Shill Bidding Dataset. The predictions related to normal and abnormal
bidding behaviour of eBay users is capable of helping companies to
identify scams and other undesirable users within the platform.

# **Data preparation**

## **Data characterisation**

Data characterisation is defined as the pre-processing phase of
summarization of the different features and characteristics presents in
the observations of a given data set, this process is typically
accomplished by introducing the data to the viewer using statistics
measure summaries, and presenting visually using graphs, like bar charts
and scatter plots (Capozzoli, Cerquitelli and Piscitelli, 2016; Han,
Kamber and Pei, 2012).

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

    ## # A tibble: 6,321 × 13
    ##    Record_ID Auction_ID Bidder_ID Bidder_Tendency Bidding_Ratio Successive_Outb…
    ##    <chr>     <chr>      <chr>               <dbl>         <dbl>            <dbl>
    ##  1 1         732        _***i              0.2            0.4                0  
    ##  2 2         732        g***r              0.0244         0.2                0  
    ##  3 3         732        t***p              0.143          0.2                0  
    ##  4 4         732        7***n              0.1            0.2                0  
    ##  5 5         900        z***z              0.0513         0.222              0  
    ##  6 8         900        i***e              0.0385         0.111              0  
    ##  7 10        900        m***p              0.4            0.222              0  
    ##  8 12        900        k***a              0.138          0.444              1  
    ##  9 13        2370       g***r              0.122          0.185              1  
    ## 10 27        600        e***t              0.155          0.346              0.5
    ## # … with 6,311 more rows, and 7 more variables: Last_Bidding <dbl>,
    ## #   Auction_Bids <dbl>, Starting_Price_Average <dbl>, Early_Bidding <dbl>,
    ## #   Winning_Ratio <dbl>, Auction_Duration <dbl>, Class <fct>

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

## **Exploratory Data Analysis**

    ## # A tibble: 6,321 × 13
    ##    Record_ID Auction_ID Bidder_ID Bidder_Tendency Bidding_Ratio Successive_Outb…
    ##    <chr>     <chr>      <chr>               <dbl>         <dbl>            <dbl>
    ##  1 1         732        _***i              0.2            0.4                0  
    ##  2 2         732        g***r              0.0244         0.2                0  
    ##  3 3         732        t***p              0.143          0.2                0  
    ##  4 4         732        7***n              0.1            0.2                0  
    ##  5 5         900        z***z              0.0513         0.222              0  
    ##  6 8         900        i***e              0.0385         0.111              0  
    ##  7 10        900        m***p              0.4            0.222              0  
    ##  8 12        900        k***a              0.138          0.444              1  
    ##  9 13        2370       g***r              0.122          0.185              1  
    ## 10 27        600        e***t              0.155          0.346              0.5
    ## # … with 6,311 more rows, and 7 more variables: Last_Bidding <dbl>,
    ## #   Auction_Bids <dbl>, Starting_Price_Average <dbl>, Early_Bidding <dbl>,
    ## #   Winning_Ratio <dbl>, Auction_Duration <dbl>, Class <fct>

    ## Warning: `gather_()` was deprecated in tidyr 1.2.0.
    ## Please use `gather()` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was generated.

![](Ebay_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

As displayed by the graphic generated by naniar function “vis_miss” on
`SBD`, the data does not include any standard missing value.

![](Ebay_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

    ## 
    ##    0    1 
    ## 5646  675

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

![](Ebay_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The heatmap above shows the correlation between all numerical features
in a single graph. The ggplot graph above has a colourblind friendly
colour palette and an added numerical scale, allowing the viewer to
check the exact correlation between the variables, as an example, the
positive correlation between early and last bidding features is
notorious. Another noticeable correlation is the negative one between
the winning ratio and auction bids, for further investigation, both is
displayed in a scatter plot.

![](Ebay_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

The scatter plot makes it noticeable why those two variables are so
positive correlated, considering that the last bidding value can never
be lower than the early bidding value. A positive slope can be found in
the diagonal centre of the 2-dimensional plot, this line represents the
values which were the same in both features.

![](Ebay_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

Different from the previous scatter plot, this time the features auction
bids and winning ratio, which had a negative correlation in the heatmap
representation, do not show any visible correlation. This interaction
was tested when used as a predictor in a linear regression method.

![](Ebay_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->![](Ebay_files/figure-gfm/unnamed-chunk-12-2.png)<!-- -->

The histograms above display a clear difference between the distribution
of the Auction and Record ID, as the x-axis of the Auction ID, is much
shorter, this happens because many auctions are represented in the data
set multiple times. Also, the Record ID displays some uniformity, even
so not perfectly uniform, representing the absence of multiple
observations with the same ID.

![](Ebay_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

The Boxplot above shows the distribution of several numerical features
with a range between 0 and 1. These 8 features were chosen to be
represented by grouped boxplots since it is an exceptional alternative
to viewing many distributions in a single graph, especially if the
y-axis range matches all features. This figure is extremely informative,
considering that it displays the distribution of the observations, the
quartile range including the median, and the minimum and maximum values,
which describe the range (Galarnyk, 2018).

![](Ebay_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

Despite being a numerical feature, the low number of distinct values
causes the histogram to not be aesthetically pleasing, however, it shows
the 5 distinct values 7,3,1,5 and 10 and how many times each one
appeared in the auction duration column. This pattern suggests five
fixed auction durations options that the platform allows the sellers to
choose from.

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

The pivot table above points to a substantial difference in the group
means value for most features when compared to normal and abnormal
bidding patterns. The class feature is the dependable variable for
machine learning models in the next steps.

## **Data cleaning**

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

    ## [1] 0

Checking for common strings associated with non-standard missing values,
the result of the analysis indicates the complete absence of them.

## **Feature engineering**

As an important stage preceding the machine learning modelling, this
task requires the judgement of the data analyst and numerous trials and
errors. Feature engineering is the act of transforming a given feature
space with different tools, especially mathematical, with the main goal
of improving the model performance (Khurana, Samulowitz and Turaga,
2018).The first step is removing unnecessary columns for the future
tasks, the first three columns only include the IDs for the record,
auction and bidder respectively and do not add any significance to the
analysis.

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
model, a new column named “Bidding difference” is substituting them with
the existing incremented value calculated by the difference between
their prices.

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

## **Data scaling**

The different orders of magnitude may bias the results since most
machine learning algorithms calculate the output using Euclidean
distance (Asaithambi, 2017). As seen in the distribution of the previous
boxplots and histograms the numeric features contain some outliers,
considering the putative relevance of these outliers in the machine
learning analysis, the ideal transformation method will maintain the
impact of these feature outliers in the output of the machine learning
algorithm to be performed. The two regularly used scaling methods are
standardization and normalization (GeeksforGeeks, 2020).

    ## No id variables; using all as measure variables

![](Ebay_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

![](Ebay_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

Since all other metrics range from 0 to 1, dividing the auction duration
variable by 10 puts it in the same range. This reduces the chance of
this variable impacting the model disproportionately compared to other
features.

As displayed before, except for a few negative values in the Bidding
difference column, all features range from 0 to 1, the data was probably
been already pre-processed. Data scaling is necessary when the features
of a dataset have a different range or order of magnitude (Roy, 2020),
which is not the case for both, taking that and the fact that the
outliers is important for the analysis, further data scaling was not
employed.

# **Dimensionality reduction**

In pattern classification using a Machine Learning algorithm, the
training vectors are evaluated by comparing them to the test split, a
common problem that can reduce its accuracy is the large dimensionality
of the data, and numerous features reducing the robustness of the
pattern classifiers (Sharma and Paliwal, 2014). The main distinction
between the two methods applied is that LDA is supervised, requiring
labelled inputs, while PCA is unsupervised, ignoring class labels. LDA
basis on discovering the feature subspace of highest separability
between classes, while PCA focus in reproduce the topmost direction of
variation in the features, the first component (PC1) accounting for the
highest variation and the PC2 the second-best and so forth (Duvva,
2021).

## **Principal Component Analysis (PCA)**

Principal component analysis or PCA is an unsupervised technique,
ignoring class labels. The main objective of performing a PCA prior to
the Machine Learning technique is to increase interpretability while
preserving statistical information (Jolliffe and Cadima, 2016).

    ## Importance of components:
    ##                           PC1    PC2    PC3     PC4     PC5     PC6     PC7
    ## Standard deviation     0.5618 0.4285 0.2541 0.24453 0.18119 0.17103 0.11862
    ## Proportion of Variance 0.4474 0.2603 0.0915 0.08476 0.04654 0.04147 0.01995
    ## Cumulative Proportion  0.4474 0.7077 0.7992 0.88399 0.93053 0.97200 0.99195
    ##                            PC8
    ## Standard deviation     0.07537
    ## Proportion of Variance 0.00805
    ## Cumulative Proportion  1.00000

The analysis resulted in 8 principal components, the first two explain
70% of the total variance. Five components were applied for the upcoming
clustering algorithms, considering that this amount is enough to
describe over 93% of the total variance (Hayden, 2018).

![](Ebay_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

Originating from the centre point of the graph above the arrows indicate
the contribution of each variable to the first two principal components.
As seen, the variables “Auction Bids” and “Starting Price Average”
contributes with high values to the first principal component, while
“Successive Outbidding” and “Bidder Tendency” to the second principal
component (Hayden, 2018).

## **Linear Discriminant Analysis (LDA)**

Another technique used to approach the high-dimensionality problem is
the Linear discriminant analysis or LDA, which different from PCA, is a
supervised technique that generally exceeds other techniques when
applied to feature classification, LDA method reduces the number of
features that fit in the different classes prior to classification while
maintaining the classes well separated in this lower dimensional space
(Sharma and Paliwal, 2014). The data was split into an 80% training, and
20% test split, this is the most common split of data considered one of
the best ranges to avoid overfitting (Askpython, 2020).

    ## Call:
    ## lda(Class ~ ., data = training_set)
    ## 
    ## Prior probabilities of groups:
    ##         0         1 
    ## 0.8932173 0.1067827 
    ## 
    ## Group means:
    ##   Bidder_Tendency Bidding_Ratio Successive_Outbidding Auction_Bids
    ## 0       0.1235555     0.1016011            0.01760018    0.2274688
    ## 1       0.3067627     0.3481041            0.83611111    0.2536599
    ##   Starting_Price_Average Winning_Ratio Auction_Duration Bidding_difference
    ## 0              0.4661744     0.3080564        0.4604383         0.02727565
    ## 1              0.5192898     0.8642052        0.4814815         0.08809889
    ## 
    ## Coefficients of linear discriminants:
    ##                               LD1
    ## Bidder_Tendency        0.05910487
    ## Bidding_Ratio          0.25399039
    ## Successive_Outbidding  7.91498217
    ## Auction_Bids           0.01319279
    ## Starting_Price_Average 0.05120990
    ## Winning_Ratio          0.30988604
    ## Auction_Duration       0.18054044
    ## Bidding_difference     0.24200412

The training data displays similar proportions of normal and abnormal
bidding behaviour found in the SBD data, 90% and 10% respectively,
including the similar group means values obtained in the exploratory
data analysis section. Represented by the Coefficients of linear
discriminants, these values indicate the predictor’s linear combination
for the different features of the data applied in the decision rule in
the LDA model. Since the class feature only contains two unique labels
the analysis results in a single linear discriminant, LD1 (Zach, 2020).

![](Ebay_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

![](Ebay_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

![](Ebay_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

LD1 does a great job separating the different classes in a single
dimension, there are almost no overlaps between normal and abnormal
bidding behaviour (Finnstats, 2021). This result suggests high
predictive accuracy in the classification algorithms that were applied
in the following sections.

# **Machine Learning**

The concept of Machine Learning is a measurable improvement in a
computer program’s function as it performance improves as it is
presented with more data. Machine Learning algorithm are built to handle
three different types of task, clustering, regression and
classification. To solve these tasks the analyst has to decide using his
experience the appropriate method to select between reinforcement
learning, supervised learning, unsupervised learning, and
semi-supervised learning, considering that each method is pertinent to
different types of data (Ray, 2019).

## **Clustering algorithms**

Clustering is a valuable set of methods used in many applications, these
unsupervised algorithms divide the features into classes, this division
is based on data patterns identified by the algorithm, and the objects
are grouped on the defined class based on similar behaviour (Dubey and
Choubey, 2017). For the following algorithms, the also unsupervised PCA
transformed data was applied, since this method resulted in a more
sparse distribution of observations that contributes to better grouping
and visualization.

### **K-Means Clustering**

This clustering algorithm is the most widely employed, the goal of
K-Means unsupervised technique is to partition the observations into k
clusters so the observations can be grouped based on the closest mean
(Dubey and Choubey, 2017).

#### **SBD K-Means Clustering**

![](Ebay_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

Elbow and silhouette methods only measure global clustering elements, as
a more refined method gap statistics formalize these methods in a
heuristic statistical in order to evaluate the ideal number of cluster
that should be applied in the k-means algorithm (Tibshirani, Walther and
Hastie, 2001).

![](Ebay_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

Setting four different clusters better identify the nature of the
features related to normal and abnormal observations. The scatter plot
displays the separation of the clusters by colour, each colour
associated with one cluster. As we know, based on the class feature,
which is not being used on unsupervised algorithms like k-means and
hierarchical agglomerative clustering. The data is supposed to be
divided into two categories, however, the normal bidding behaviour has a
way bigger proportion than abnormal behaviour, thus being subjected to
multiple patterns within itself (Maklin, 2019).

#### **PCA transformed K-Means Clustering**

![](Ebay_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

The ideal number of clusters for LDA transformed data is also four, as
displayed in the gap statistical method above (Tibshirani, Walther and
Hastie, 2001).

![](Ebay_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

As exhibited in the scatter plot above the PCA transformed data achieve
a fantastic job dividing the data into 4 different clusters, mainly on
the x-axis.

### **Hierarchical Agglomerative Clustering**

As with other multivariate methods, the implemented hierarchical
agglomerative clustering algorithm is an essential tool for the embedded
classification of numerical features using sets of variables. This
method is usually applied when the data has a large number of partitions
each one correlated with a level of the hierarchy (Murtagh and
Contreras, 2017).

#### **SBD Hierarchical Agglomerative Clustering**

![](Ebay_files/figure-gfm/unnamed-chunk-33-1.png)<!-- -->

The dendrogram graphic representation above represents the number of
partitions within the data. Asobserved, the ideal number of clusters is
also four, as obtained in the k-means section, this is noticeable by
inspecting the vertical axis (Maklin, 2019).

#### **PCA transformed Hierarchical Agglomerative Clustering**

![](Ebay_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

Both K-Means and Hierarchical Agglomerative Clustering exhibited similar
results, indicating four patterns, being the normal data with three
different patterns within itself and the abnormal data in a different
fourth fashion, as suggested by the respective diagnostic plots. Both
methods did an amazing job identifying and discriminating the different
categories.

## **Classification algorithms**

Applied in both structured and unstructured data, classification
algorithms aim is to learn from training data in order to label new data
into a specific class (Sen, Hajra and Ghosh, 2019).

### **Random Forest Classification**

The Random Forest Classification technique can be applied in both
regression and classification, this supervised learning methodology
performs well with numerous and imbalanced datasets, dealing with
missing data efficiently (More and Rana, 2017).

#### **SBD Random Forest Classification**

    ## 
    ## Call:
    ##  randomForest(formula = Class ~ ., data = training_set, importance = TRUE,      proximity = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.73%
    ## Confusion matrix:
    ##      0   1 class.error
    ## 0 4495  22 0.004870489
    ## 1   15 525 0.027777778

![](Ebay_files/figure-gfm/unnamed-chunk-37-1.png)<!-- -->

The non-transformed data were fitted in the random forest classifier
method and resulted in an estimated error rate of 0.65% after running
500 decision trees, which is extremely good. The confusion matrix shows
the superb predictive power of the model.However, The fact that not
every good and precise model performs as accurate in real-world
scenarios should be considered (Kumar, 2020). The same code was adapted
and applied to the data transformed by the LDA method, and the results
compared.

#### **LDA transformed Random Forest Classification**

    ## 
    ## Call:
    ##  randomForest(formula = class ~ ., data = training_set_lda, importance = TRUE,      proximity = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 1
    ## 
    ##         OOB estimate of  error rate: 0%
    ## Confusion matrix:
    ##      0   1 class.error
    ## 0 4379   0           0
    ## 1    0 678           0

![](Ebay_files/figure-gfm/unnamed-chunk-39-1.png)<!-- -->

The accuracy score comparison indicates the LDA transformed data as the
best approach, resulting in an amazing 0% error rate. The confusion
matrix displays that the model is 100% precise in identifying normal
behaviour in both cases, which is expected considering that the data
contains almost 10 times more normal than abnormal. A recommendation for
future analysis would be collect data with more abnormal observations to
make the model even more reliable (Kohli, 2019).

### **K-Nearest Neighbours**

Mostly used as a classifier, K-Nearest Neighbours or KNN method do its
calculations using euclidean distance. This supervised non-parametric
method is known for its simplicity and efficacy (Taunk et al., 2019).

#### **SBD K-Nearest Neighbours**

    ##    classifier_knn
    ##        0    1
    ##   0 1129    0
    ##   1    0  135

    ## [1] "SBD Accuracy = 1"

#### **LDA transformed K-Nearest Neighbours**

    ##    classifier_knn_lda
    ##        0    1
    ##   0 1105    0
    ##   1    0  159

    ## [1] "LDA transformed SBD Accuracy = 1"

The K-Nearest Neighbours prediction using Euclidean distance is
displayed by the confusion matrix with 100% precision in both
transformed and non-transformed data, this 100% accuracy score is
similar to the one obtained in the previous method.

### **Decision Tree**

Another supervised method, decision tree method combines a sequence of
straightforward techniques, in which each one makes use of a threshold
value as a metric against possible values, this method classifies the
data that fall in a particular region as integrated to the most
recurrent class in that region. The decision tree’s main goal is to
depict a tree with its findings data patterns accomplished by checking
which test best separate the different cases into classes (Kotsiantis,
2011).

#### **SBD Decision tree**

![](Ebay_files/figure-gfm/unnamed-chunk-42-1.png)<!-- -->

The decision tree plot above exhibits on its nodes insights about the
data, the first one reiterates that the data is composed of only 11%
abnormal bidding behaviour. Following the track, the decision tree asks
if successive outbidding is inferior to 0.25, if it is true it directs
to the left root, which displays the fact that 87% of the normal bidding
behaviour falls in this category (Johnson, 2022).

    ## 
    ## Classification tree:
    ## rpart(formula = Class ~ ., data = training_set)
    ## 
    ## Variables actually used in tree construction:
    ## [1] Auction_Duration      Successive_Outbidding Winning_Ratio        
    ## 
    ## Root node error: 540/5057 = 0.10678
    ## 
    ## n= 5057 
    ## 
    ##         CP nsplit rel error   xerror     xstd
    ## 1 0.727778      0  1.000000 1.000000 0.040671
    ## 2 0.090741      1  0.272222 0.272222 0.022124
    ## 3 0.040123      2  0.181481 0.181481 0.018154
    ## 4 0.010000      5  0.061111 0.061111 0.010603

![](Ebay_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 4506   22
    ##          1   11  518
    ##                                           
    ##                Accuracy : 0.9935          
    ##                  95% CI : (0.9908, 0.9955)
    ##     No Information Rate : 0.8932          
    ##     P-Value [Acc > NIR] : < 2e-16         
    ##                                           
    ##                   Kappa : 0.9655          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.08172         
    ##                                           
    ##             Sensitivity : 0.9976          
    ##             Specificity : 0.9593          
    ##          Pos Pred Value : 0.9951          
    ##          Neg Pred Value : 0.9792          
    ##              Prevalence : 0.8932          
    ##          Detection Rate : 0.8910          
    ##    Detection Prevalence : 0.8954          
    ##       Balanced Accuracy : 0.9784          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

Amazing 99.4% accuracy obtained, the confusion matrix only reported nine
false negatives, and twenty false positives in 5057 observation.

    ## Setting direction: controls < cases

![](Ebay_files/figure-gfm/unnamed-chunk-45-1.png)<!-- -->

ROC curves can be employed to distinguish the balance of sensitivity and
specificity when dealing with binary classifiers, the result can be
interpreted as a measure of the predictive robustness of a model
(Horton, 2016). As expected from the previous results the ROC curve
displays a result close to perfection, the bidding behaviour is indeed
very predictable as observed in the true positive rate (y-axis) versus
false positive rate (x-axis) as also shown in the confusion matrix.

#### **LDA transformed Decision tree**

![](Ebay_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->

The single dimensionality of the LDA transformed data allows the
decision tree to cover the complexity of the data set in a unique
partitioning

    ## 
    ## Classification tree:
    ## rpart(formula = class ~ ., data = training_set_lda)
    ## 
    ## Variables actually used in tree construction:
    ## [1] LD1
    ## 
    ## Root node error: 678/5057 = 0.13407
    ## 
    ## n= 5057 
    ## 
    ##     CP nsplit rel error xerror     xstd
    ## 1 1.00      0         1      1 0.035738
    ## 2 0.01      1         0      0 0.000000

![](Ebay_files/figure-gfm/unnamed-chunk-47-1.png)<!-- -->

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 4379    0
    ##          1    0  678
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9993, 1)
    ##     No Information Rate : 0.8659     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ##                                      
    ##             Sensitivity : 1.0000     
    ##             Specificity : 1.0000     
    ##          Pos Pred Value : 1.0000     
    ##          Neg Pred Value : 1.0000     
    ##              Prevalence : 0.8659     
    ##          Detection Rate : 0.8659     
    ##    Detection Prevalence : 0.8659     
    ##       Balanced Accuracy : 1.0000     
    ##                                      
    ##        'Positive' Class : 0          
    ## 

    ## Setting direction: controls < cases

![](Ebay_files/figure-gfm/unnamed-chunk-49-1.png)<!-- -->

The LDA transformed data resulted in perfect predictions, reporting even
better results than the already exceptional obtained from the
untransformed data.

### **Comparison of ML algorithms for Classification**

Every single supervised algorithm resulted in an outstanding
performance, Random Forest Classification, K-Nearest Neighbours and
Decision Tree resulted in an accuracy score of over 98% on SBD
predictions and 100% on LDA transformed data, with minor differences
between them. K-Nearest Neighbours had the best accuracy score, however,
the difference is negligible, it is safe to say that all algorithms
applied are suitable for predictions on SBD.

# **Conclusion**

As a result of the dimensionality reduction methods, it was observed
that the PCA empowered the clustering methods with a higher level of
discrimination between groups. Scaled PCA resulted in more sparsed
observations in the two-dimensional plot. The three classification
methods employed in this report had magnificent results, especially
K-Nearest Neighbours, achieving perfect predictions in both
LDA-transformed and untransformed data. The LDA methods resulted in a
more versatile one dimensionaç analysis, with a clear separation between
classes, easily observed in diagnostic graphics.

Three supervised algorithms were implemented for classification
purposes, Random Forest, K-Nearest Neighbours and Decision Tree. Every
single one of these algorithms resulted in an outstanding accuracy
score, being excellent in extracting predictions for the normal and
abnormal bidding behaviour, exceptionally normal, because of the massive
amount of observation that falls into this category. Two clustering
methods were applied for the unsupervised algorithms using the PCA
transformed data, K-Means Clustering and Hierarchical Agglomerative
Clustering. It was observed in both algorithms clear division of the
data into four distinct patterns, three associated with normal bidding
behaviour and one with abnormal.

A further step to improve the prediction power of these models for the
detection of abnormal behaviour, putatively associated with scams and
other unwanted conduct, could be adding more data with feature values
associated with the abnormal bidding patterns observed in the algorithms
implemented. Considering the results, it is conclusive that all
implemented algorithms performed extremely well, especially K-nearest
neighbours regression with feature reduced data by LDA.

# **References**

Alzahrani, A. and Sadaoui, S. (2018). Scraping and preprocessing
commercial auction data for fraud classification.

Analytics Vidhya. (2021). Tackling Missing Value in Dataset. \[online\]
Available at: <a
href="https://www.analyticsvidhya.com/blog/2021/10/handlingmissing-value/#"
class="uri">https://www.analyticsvidhya.com/blog/2021/10/handlingmissing-value/#</a>:\~:text=Missing%20value%20can%20bias%20the
\[Accessed 28 Apr. 2022\].

Asaithambi, S. (2017). Why, How and When to Scale your Features.
\[online\] Medium. Available at: <a
href="https://medium.com/greyatom/why-howand-when-to-scale-your-features-4b30ab09db5e"
class="uri">https://medium.com/greyatom/why-howand-when-to-scale-your-features-4b30ab09db5e</a>.

Askpython (2020). Split Training and Testing Data Sets in Python -
AskPython. \[online\] Available at:
<https://www.askpython.com/python/examples/split-data-training-and-testing-set#>:\~:text=The%20most%20common%20split%20ratio
\[Accessed 19 May 2022\].

Capozzoli, A., Cerquitelli, T. and Piscitelli, M.S. (2016). Enhancing
energy efficiency in buildings through innovative data analytics
technologiesa. Pervasive Computing, pp.353–389

Dubey, A. and Choubey, A.P.D.A., 2017. A systematic review on k-means
clustering techniques. Int J Sci Res Eng Technol (IJSRET, ISSN
2278–0882), 6(6).

Duvva, P. (2021). LDA vs. PCA – Towards AI. \[online\] Available at:
<a href="https://towardsai.net/p/data-science/lda-vspca#"
class="uri">https://towardsai.net/p/data-science/lda-vspca#</a>:\~:text=LDA%20focuses%20on%20finding%20

Finnstats (2021). Linear Discriminant Analysis in R \| R-bloggers.
\[online\] Available at:
<https://www.r-bloggers.com/2021/05/linear-discriminant-analysis-in-r/>.

Galarnyk, M. (2018). Understanding Boxplots. \[online\] Medium.
Available at: <a
href="https://towardsdatascience.com/understanding-boxplots5e2df7bcbd51"
class="uri">https://towardsdatascience.com/understanding-boxplots5e2df7bcbd51</a>.

GeeksforGeeks. (2020). Normalization vs Standardization. \[online\]
Available at: <a
href="https://www.geeksforgeeks.org/normalization-vsstandardization/"
class="uri">https://www.geeksforgeeks.org/normalization-vsstandardization/</a>
\[Accessed 1 May 2022\].

Han, J., Kamber, M. and Pei, J. (2012). Introduction. Data Mining,
pp.1–38.

Hayden, L.(2018). R PCA Tutorial (Principal Component Analysis).
\[online\] Available at:
<https://www.datacamp.com/tutorial/pca-analysis-r> \[Accessed 16
Jun. 2022\].

Horton, B.(2016). ROC Curves in Two Lines of R Code. \[online\]
Available at:
<https://blog.revolutionanalytics.com/2016/08/roc-curves-in-two-lines-of-code.html#>:\~:text=ROC%20curves%20are%20commonly%20used
\[Accessed 20 Jun. 2022\].

Hunt, C. (2019). Why you shouldn’t use pie charts. \[online\]
Statistical Consulting Centre. Available at: <a
href="https://scc.ms.unimelb.edu.au/resources-list/data-visualisation-and-exploration/no_piecharts#"
class="uri">https://scc.ms.unimelb.edu.au/resources-list/data-visualisation-and-exploration/no_piecharts#</a>:\~:text=Pies%20and%20doughnuts%20fail%20because
\[Accessed 27 Apr. 2022\]

Johnson, D.(2022). Decision Tree in R with Example. \[online\] Available
at: <https://www.guru99.com/r-decision-trees.html>.

Jolliffe, I.T. and Cadima, J. (2016). Principal component analysis: a
review and recent developments. Philosophical Transactions of the Royal
Society A: Mathematical, Physical and Engineering Sciences, 374(2065),
p.20150202. <doi:10.1098/rsta.2015.0202>.

Khurana, U., Samulowitz, H. and Turaga, D. (2018) “Feature Engineering
for Predictive Modeling Using Reinforcement Learning”, Proceedings of
the AAAI Conference on Artificial Intelligence, 32(1). Available at:
<https://ojs.aaai.org/index.php/AAAI/article/view/11678>

Kohli, S. (2019). Understanding a Classification Report For Your Machine
Learning Model. \[online\] Medium. Available at:
<https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397>.

Kotsiantis, S.B. (2011). Decision trees: a recent overview. Artificial
Intelligence Review, 39(4), pp.261–283. <doi:10.1007/s10462-011-9272-4>.

Kumar, S. (2020). 9 Reasons why Machine Learning models not perform well
in production. \[online\] Medium. Available at:
<https://towardsdatascience.com/9-reasons-why-machine-learning-models-not-perform-well-in-production-4497d3e3e7a5>
\[Accessed 20 May 2022\].

Maklin, C. (2019). K-means Clustering Python Example. \[online\] Medium.
Available at: <a
href="https://towardsdatascience.com/machine-learningalgorithms-part-9-k-means-example-in-python-f2ad05ed5203"
class="uri">https://towardsdatascience.com/machine-learningalgorithms-part-9-k-means-example-in-python-f2ad05ed5203</a>.

More, A.S. and Rana, D.P. (2017). Review of random forest classification
techniques to resolve data imbalance. 2017 1st International Conference
on Intelligent Systems and Information Management (ICISIM).
<doi:10.1109/icisim.2017.8122151>.

Murtagh, F. and Contreras, P. (2017). Algorithms for hierarchical
clustering: an overview, II. Wiley Interdisciplinary Reviews: Data
Mining and Knowledge Discovery, 7(6), p.e1219. <doi:10.1002/widm.1219>.

Ray, S. (2019). A Quick Review of Machine Learning Algorithms.
\[online\] IEEE Xplore. <doi:10.1109/COMITCon.2019.8862451>.

Sen, P.C., Hajra, M. and Ghosh, M. (2019). Supervised Classification
Algorithms in Machine Learning: A Survey and Review. Advances in
Intelligent Systems and Computing, pp.99–111.
<doi:10.1007/978-981-13-7403-6_11>.

Sharma, A. and Paliwal, K.K. (2014). Linear discriminant analysis for
the small sample size problem: an overview. International Journal of
Machine Learning and Cybernetics, 6(3), pp.443–454.
<doi:10.1007/s13042-013-0226-9>.

Sullivan, J. (2018). Data Cleaning with Python and Pandas: Detecting
Missing Values. \[online\] Medium. Available at:
<https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b>.

Taunk, K., De, S., Verma, S. and Swetapadma, A. (2019). A Brief Review
of Nearest Neighbor Algorithm for Learning and Classification.
\[online\] IEEE Xplore. <doi:10.1109/ICCS45141.2019.9065747>.

Tibshirani, R., Walther, G. and Hastie, T. (2001). Estimating the number
of clusters in a data set via the gap statistic. Journal of the Royal
Statistical Society: Series B (Statistical Methodology), \[online\]
63(2), pp.411–423. <doi:10.1111/1467-9868.00293>.

Zach (2020). Linear Discriminant Analysis in R (Step-by-Step).
\[online\] Statology. Available at:
<https://www.statology.org/linear-discriminant-analysis-in-r/>.
