# HierarPy [in progress]

Tools for calculating dominance hiearchies in Python. Working on including:

- Elo Scores [Albers & de Vries 2001](https://www.sciencedirect.com/science/article/pii/S0003347200915719)
- David's Scores [David 1987](https://academic.oup.com/biomet/article-abstract/74/2/432/239730?redirectedFrom=fulltext)
- ADAGIO [Douglas *et al* 2018](https://www.sciencedirect.com/science/article/pii/S0003347216302639?via%3Dihub)
- Randomized Elo [Sánchez‐Tójar *et al* 2017](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.12776)
- Linearity Statistics

# Usage:

First, import packages:

```python
import pandas as pd
import hierarpy as hp
```

Load an example dataframe (provided in `hierarpy/test_dfs`)

```python
df = pd.read_csv('test_dfs/df1.csv')
```

This dataframe looks like the following:

|    | datetime            | winner   | loser   | sex_winner   | sex_loser   |
|---:|:--------------------|:---------|:--------|:-------------|:------------|
|  0 | 2016-09-08 12:19:41 | A        | G       | m            | m           |
|  1 | 2016-09-08 12:24:35 | A        | C       | m            | m           |
|  2 | 2016-09-08 14:43:32 | B        | C       | m            | m           |
|  3 | 2016-09-08 15:26:44 | C        | B       | m            | m           |
|  4 | 2016-09-08 17:08:47 | C        | R       | m            | m           |

## Processing interactions into a matrix:

We can tabulate this dataframe using hierarpy's `matrix_from_dataframe` function:

    mat = hp.matrix_from_dataframe(df, Winner = 'winner', Loser = 'loser')

Resulting in the following matrix of interactions:

| winner   |   A |   B |   C |   D |   E |   F |   G |   H |   I |   J |   K |   L |   M |   N |   O |   P |   Q |   R |   S |   T |   U |   V |   W |   X |   Y |   Z |
|:---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| A        |   0 |   2 |  11 |   1 |   1 |   0 |   6 |   0 |   0 |   0 |   1 |   2 |   0 |   0 |   5 |   2 |   1 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |
| B        |   0 |   0 |   2 |   0 |   1 |   1 |   0 |   0 |   0 |   0 |   1 |   1 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |
| C        |   2 |   3 |   0 |   1 |   0 |   0 |   3 |   0 |   0 |   1 |   1 |   0 |   1 |   1 |   0 |   2 |   0 |   1 |   1 |   3 |   3 |   0 |   1 |   1 |   1 |   0 |
| D        |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| E        |   0 |   0 |   2 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   2 |   0 |   0 |   0 |   0 |   1 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   1 |
| F        |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |
| G        |   0 |   1 |   3 |   1 |   0 |   2 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   3 |   0 |   0 |   0 |   2 |   0 |   1 |   1 |   1 |   0 |   6 |   1 |
| H        |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| I        |   1 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   3 |   0 |   0 |   0 |   0 |   2 |   1 |   0 |   0 |   0 |   0 |   1 |   0 |
| J        |   0 |   0 |   2 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   6 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |
| K        |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   3 |   1 |   0 |   1 |   1 |   0 |   0 |   0 |   0 |   1 |   1 |   0 |
| L        |   0 |   0 |   2 |   0 |   0 |   0 |   2 |   0 |   0 |   0 |   0 |   0 |   0 |   3 |   5 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| M        |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   1 |   0 |   0 |   0 |   2 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| N        |   7 |   4 |   5 |   2 |   0 |   3 |   5 |   0 |   3 |   1 |   3 |   2 |   1 |   1 |   6 |   3 |   0 |   1 |   1 |   1 |   0 |   3 |   1 |   0 |   0 |   2 |
| O        |   0 |   0 |   2 |   2 |   0 |   0 |   0 |   0 |   1 |   0 |   1 |   0 |   0 |   2 |   0 |   1 |   0 |   2 |   0 |   3 |   0 |   0 |   0 |   0 |   1 |   0 |
| P        |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   1 |   0 |   1 |   2 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| Q        |   0 |   0 |   1 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| R        |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| S        |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| T        |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   2 |   0 |
| U        |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   3 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |
| V        |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| W        |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   2 |   0 |
| X        |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |
| Y        |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| Z        |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |


## Getting David's ranks and scores

We can get David's ranks and scores from such a matrix using the function `david_ranks`:

    david = hp.david_ranks(mat)

And david will now be a dataframe with scores and ranks for each individual:

|    | ind   |    D_score |   Davids_rank |
|---:|:------|-----------:|--------------:|
|  1 | N     |  67.5      |             1 |
|  2 | A     |  53.2454   |             2 |
|  3 | Q     |  35.3929   |             3 |
|  4 | J     |  29.9515   |             4 |
|  5 | H     |  28.3057   |             5 |
|  6 | E     |  25.1667   |             6 |
|  7 | L     |  24.0262   |             7 |
|  8 | G     |  16.0595   |             8 |
|  9 | I     |  15.789    |             9 |
| 10 | B     |  13.2095   |            10 |
| 11 | M     |   3.96337  |            11 |
| 12 | C     |   0.341026 |            12 |
| 13 | V     |  -4.79048  |            13 |
| 14 | P     |  -5.50714  |            14 |
| 15 | O     |  -7.44048  |            15 |
| 16 | F     |  -8.7      |            16 |
| 17 | K     | -12.2667   |            17 |
| 18 | Z     | -15.3738   |            18 |
| 19 | U     | -16.8295   |            19 |
| 20 | X     | -17.2462   |            20 |
| 21 | W     | -24.22     |            21 |
| 22 | D     | -28.4405   |            22 |
| 23 | T     | -28.6071   |            23 |
| 24 | S     | -39.87     |            24 |
| 25 | R     | -50.1866   |            25 |
| 26 | Y     | -53.4723   |            26 |

# Getting ADAGIO graph and ranks

We can get the ADAGIO graph's nodes and edges from a matrix using the function `run_ADAGIO`. This function can take the arguments `preprocess_data` (Boolean, see paper for details), and `plot` (Also Boolean, whether or not to plot the resulting graph).

    nodes, edges = hp.run_ADAGIO(mat, preprocess_data = False ,plot=True)

This results in the following plot:

![ADAGIO graph](https://github.com/sacul-git/hierarpy/blob/master/hierarpy/ADAGIO_graph.png)

From these nodes and edges, we can convert to rankings using the function `rank_from_graph`. We can use the argument `method` to choose "bottom-up" or "top-down" rankings (see paper for details):

|    | ind   |   adagio_rank |
|---:|:------|--------------:|
| 12 | M     |             1 |
| 23 | X     |             1 |
| 22 | W     |             1 |
|  4 | E     |             1 |
| 16 | Q     |             1 |
|  7 | H     |             1 |
|  8 | I     |             1 |
|  9 | J     |             1 |
| 11 | L     |             1 |
| 13 | N     |             2 |
| 21 | V     |             3 |
| 15 | P     |             3 |
|  0 | A     |             3 |
| 10 | K     |             3 |
|  6 | G     |             4 |
|  2 | C     |             4 |
| 14 | O     |             5 |
|  5 | F     |             5 |
| 20 | U     |             5 |
|  1 | B     |             5 |
| 25 | Z     |             5 |
| 17 | R     |             6 |
| 19 | T     |             6 |
|  3 | D     |             6 |
| 18 | S     |             7 |
| 24 | Y     |             7 |

## More to come :)