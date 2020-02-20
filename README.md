# Big-Data-Analytics
Technical project
Big Data Analytics Technical Project
L00151287 Anitmon Baby kanakkalil

Preprocessing
#### Importing needful libraries

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer

Show code
##Loading CSV data
dataset = "/FileStore/tables/train.csv"
titanic_df = spark.read.csv(dataset,header = 'True',inferSchema='True')
display(titanic_df)
147	1	3	"Andersson, Mr. August Edvard (""Wennerstrom"")"	male	27	0	0	350043	7.7958	null	S
519	1	2	"Angle, Mrs. William A (Florence ""Mary"" Agnes Hughes)"	female	36	1	0	226875	26	null	S
291	1	1	"Barber, Miss. Ellen ""Nellie"""	female	26	0	0	19877	78.85	null	S
625	0	3	"Bowen, Mr. David John ""Dai"""	male	21	0	0	54636	16.1	null	S
508	1	1	"Bradley, Mr. George (""George Arthur Brayton"")"	male	null	0	0	111427	26.55	null	S
346	1	2	"Brown, Miss. Amelia ""Mildred"""	female	24	0	0	248733	13	F33	S
209	1	3	"Carr, Miss. Helen ""Ellen"""	female	16	0	0	367231	7.75	null	Q
205	1	3	"Cohen, Mr. Gurshon ""Gus"""	male	18	0	0	A/5 3540	8.05	null	S
238	1	2	"Collyer, Miss. Marjorie ""Lottie"""	female	8	0	2	C.A. 31921	26.25	null	S
490	1	3	"Coutts, Master. Eden Leslie ""Neville"""	male	9	1	1	C.A. 37671	15.9	null	S
349	1	3	"Coutts, Master. William Loch ""William"""	male	3	1	1	C.A. 37671	15.9	null	S
557	1	1	"Duff Gordon, Lady. (Lucille Christiana Sutherland) (""Mrs Morgan"")"	female	48	1	0	11755	39.6	A16	C
600	1	1	"Duff Gordon, Sir. Cosmo Edmund (""Mr Morgan"")"	male	49	1	0	PC 17485	56.9292	A20	C
573	1	1	"Flynn, Mr. John Irwin (""Irving"")"	male	36	0	0	PC 17474	26.3875	E25	S
437	0	3	"Ford, Miss. Doolina Margaret ""Daisy"""	female	21	2	2	W./C. 6608	34.375	null	S
148	0	3	"Ford, Miss. Robina Maggie ""Ruby"""	female	9	2	2	W./C. 6608	34.375	null	S
482	0	2	"Frost, Mr. Anthony Wood ""Archie"""	male	null	0	0	239854	0	null	S
157	1	3	"Gilnagh, Miss. Katherine ""Katie"""	female	16	0	0	35851	7.7333	null	Q
166	1	3	"Goldsmith, Master. Frank John William ""Frankie"""	male	9	0	2	363291	20.525	null	S
721	1	2	"Harper, Miss. Annie Jessie ""Nina"""	female	6	0	1	248727	33	null	S
275	1	3	"Healy, Miss. Hanora ""Nora"""	female	null	0	0	370375	7.75	null	Q
655	0	3	"Hegarty, Miss. Hanora ""Nora"""	female	18	0	0	365226	6.75	null	Q
605	1	1	"Homer, Mr. Harry (""Mr E Haven"")"	male	35	0	0	111426	26.55	null	C
889	0	3	"Johnston, Miss. Catherine Helen ""Carrie"""	female	null	1	2	W./C. 6607	23.45	null	S
791	0	3	"Keane, Mr. Andrew ""Andy"""	male	null	0	0	12460	7.75	null	Q
301	1	3	"Kelly, Miss. Anna Katherine ""Annie Kate"""	female	null	0	0	9234	7.75	null	Q
707	1	2	"Kelly, Mrs. Florence ""Fannie"""	female	45	0	0	223596	13.5	null	S
554	1	3	"Leeni, Mr. Fahim (""Philip Zenni"")"	male	22	0	0	2620	7.225	null	C
228	0	3	"Lovell, Mr. John Hall (""Henry"")"	male	20.5	0	0	A/5 21173	7.25	null	S
199	1	3	"Madigan, Miss. Margaret ""Maggie"""	female	null	0	0	370370	7.75	null	Q
711	1	1	"Mayne, Mlle. Berthe Antonine (""Mrs de Villiers"")"	female	24	0	0	PC 17482	49.5042	C90	C
23	1	3	"McGowan, Miss. Anna ""Annie"""	female	15	0	0	330923	8.0292	null	Q
360	1	3	"Mockler, Miss. Helen Mary ""Ellie"""	female	null	0	0	330980	7.8792	null	Q
706	0	2	"Morley, Mr. Henry Samuel (""Mr Henry Marshall"")"	male	39	0	0	250655	26	null	S
710	1	3	"Moubarek, Master. Halim Gonios (""William George"")"	male	null	1	1	2661	15.2458	null	C
698	1	3	"Mullens, Miss. Katherine ""Katie"""	female	null	0	0	35852	7.7333	null	Q
242	1	3	"Murphy, Miss. Katherine ""Kate"""	female	null	1	0	367230	15.5	null	Q
876	1	3	"Najib, Miss. Adele Kiamie ""Jane"""	female	15	0	0	2667	7.225	null	C
382	1	3	"Nakid, Miss. Maria (""Mary"")"	female	1	0	2	2653	15.7417	null	C
149	0	2	"Navratil, Mr. Michel (""Louis M Hoffman"")"	male	36.5	0	2	230080	26	F2	S
187	1	3	"O'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)"	female	null	1	0	370365	15.5	null	Q
29	1	3	"O'Dwyer, Miss. Ellen ""Nellie"""	female	null	0	0	330959	7.8792	null	Q
654	1	3	"O'Leary, Miss. Hanora ""Norah"""	female	null	0	0	330919	7.8292	null	Q
278	0	2	"Parkes, Mr. Francis ""Frank"""	male	null	0	0	239853	0	null	S
102	0	3	"Petroff, Mr. Pastcho (""Pentcho"")"	male	null	0	0	349215	7.8958	null	S
428	1	2	"Phillips, Miss. Kate Florence (""Mrs Kate Louise Phillips Marshall"")"	female	19	0	0	250655	26	null	S
188	1	1	"Romaine, Mr. Charles Hallace (""Mr C Rolmane"")"	male	45	0	0	111428	26.55	null	S
743	1	1	"Ryerson, Miss. Susan Parker ""Suzette"""	female	21	2	2	PC 17608	262.375	B57 B59 B63 B66	C
864	0	3	"Sage, Miss. Dorothy Edith ""Dolly"""	female	null	8	2	CA. 2343	69.55	null	S
718	1	2	"Troutt, Miss. Edwina Celia ""Winnie"""	female	27	0	0	34218	10.5	E101	S
162	1	2	"Watt, Mrs. James (Elizabeth ""Bessie"" Inglis Milne)"	female	40	0	0	C.A. 33595	15.75	null	S
305	0	3	"Williams, Mr. Howard Hugh ""Harry"""	male	null	0	0	A/5 2466	8.05	null	S
200	0	2	"Yrois, Miss. Henriette (""Mrs Harbeck"")"	female	24	0	0	248747	13	null	S
846	0	3	Abbing, Mr. Anthony	male	42	0	0	C.A. 5547	7.55	null	S
747	0	3	Abbott, Mr. Rossmore Edward	male	16	1	1	C.A. 2673	20.25	null	S
280	1	3	Abbott, Mrs. Stanton (Rosa Hunt)	female	35	1	1	C.A. 2673	20.25	null	S
309	0	2	Abelson, Mr. Samuel	male	30	1	0	P/PP 3381	24	null	C
875	1	2	Abelson, Mrs. Samuel (Hannah Wizosky)	female	28	1	0	P/PP 3381	24	null	C
366	0	3	Adahl, Mr. Mauritz Nils Martin	male	30	0	0	C 7076	7.25	null	S
402	0	3	Adams, Mr. John	male	26	0	0	341826	8.05	null	S
41	0	3	Ahlin, Mrs. Johan (Johanna Persdotter Larsson)	female	40	1	0	7546	9.475	null	S
856	1	3	Aks, Mrs. Sam (Leah Rosen)	female	18	0	1	392091	9.35	null	S
208	1	3	Albimona, Mr. Nassef Cassem	male	26	0	0	2699	18.7875	null	C
811	0	3	Alexander, Mr. William	male	26	0	0	3474	7.8875	null	S
841	0	3	Alhomaki, Mr. Ilmari Rudolf	male	20	0	0	SOTON/O2 3101287	7.925	null	S
211	0	3	Ali, Mr. Ahmed	male	24	0	0	SOTON/O.Q. 3101311	7.05	null	S
785	0	3	Ali, Mr. William	male	25	0	0	SOTON/O.Q. 3101312	7.05	null	S
731	1	1	Allen, Miss. Elisabeth Walton	female	29	0	0	24160	211.3375	B5	S
5	0	3	Allen, Mr. William Henry	male	35	0	0	373450	8.05	null	S
306	1	1	Allison, Master. Hudson Trevor	male	0.92	1	2	113781	151.55	C22 C26	S
298	0	1	Allison, Miss. Helen Loraine	female	2	1	2	113781	151.55	C22 C26	S
499	0	1	Allison, Mrs. Hudson J C (Bessie Waldo Daniels)	female	25	1	2	113781	151.55	C22 C26	S
835	0	3	Allum, Mr. Owen George	male	18	0	0	2223	8.3	null	S
193	1	3	Andersen-Jensen, Miss. Carla Christine Nielsine	female	19	1	0	350046	7.8542	null	S
461	1	1	Anderson, Mr. Harry	male	48	0	0	19952	26.55	E12	S
851	0	3	Andersson, Master. Sigvard Harald Elias	male	4	4	2	347082	31.275	null	S
814	0	3	Andersson, Miss. Ebba Iris Alfrida	female	6	4	2	347082	31.275	null	S
120	0	3	Andersson, Miss. Ellis Anna Maria	female	2	4	2	347082	31.275	null	S
69	1	3	Andersson, Miss. Erna Alexandra	female	17	4	2	3101281	7.925	null	S
542	0	3	Andersson, Miss. Ingeborg Constanzia	female	9	4	2	347082	31.275	null	S
543	0	3	Andersson, Miss. Sigrid Elisabeth	female	11	4	2	347082	31.275	null	S
14	0	3	Andersson, Mr. Anders Johan	male	39	1	5	347082	31.275	null	S
611	0	3	Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren)	female	39	1	5	347082	31.275	null	S
92	0	3	Andreasson, Mr. Paul Edvin	male	20	0	0	347466	7.8542	null	S
145	0	2	Andrew, Mr. Edgardo Samuel	male	18	0	0	231945	11.5	null	S
276	1	1	Andrews, Miss. Kornelia Theodosia	female	63	1	0	13502	77.9583	D7	S
807	0	1	Andrews, Mr. Thomas Jr	male	39	0	0	112050	0	A36	S
572	1	1	Appleton, Mrs. Edward Dale (Charlotte Lamson)	female	53	2	0	11769	51.4792	C101	S
354	0	3	Arnold-Franchi, Mr. Josef	male	25	1	0	349237	17.8	null	S
50	0	3	Arnold-Franchi, Mrs. Josef (Josefine Franchi)	female	18	1	0	349237	17.8	null	S
494	0	1	Artagaveytia, Mr. Ramon	male	71	0	0	PC 17609	49.5042	null	C
364	0	3	Asim, Mr. Adola	male	35	0	0	SOTON/O.Q. 3101310	7.05	null	S
183	0	3	Asplund, Master. Clarence Gustaf Hugo	male	9	4	2	347077	31.3875	null	S
262	1	3	Asplund, Master. Edvin Rojj Felix	male	3	4	2	347077	31.3875	null	S
234	1	3	Asplund, Miss. Lillian Gertrud	female	5	4	2	347077	31.3875	null	S
26	1	3	Asplund, Mrs. Carl Oscar (Selma Augusta Emilia Johansson)	female	38	1	5	347077	31.3875	null	S
701	1	1	Astor, Mrs. John Jacob (Madeleine Talmadge Force)	female	18	1	0	PC 17757	227.525	C62 C64	C
115	0	3	Attalah, Miss. Malake	female	17	0	0	2627	14.4583	null	C
245	0	3	Attalah, Mr. Sleiman	male	30	0	0	2694	7.225	null	C
370	1	1	Aubart, Mme. Leontine Pauline	female	24	0	0	PC 17477	69.3	B35	C
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
Analyzing the dataset
###The below is the schema of data
titanic_df.printSchema()
root
 |-- PassengerId: integer (nullable = true)
 |-- Survived: integer (nullable = true)
 |-- Pclass: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: double (nullable = true)
 |-- SibSp: integer (nullable = true)
 |-- Parch: integer (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: double (nullable = true)
 |-- Cabin: string (nullable = true)
 |-- Embarked: string (nullable = true)

passengers_count = titanic_df.count()
print(passengers_count)
891
###Viewing few rows
titanic_df.show(5)
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
|PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
|          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|
|          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
|          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|
|          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|
|          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| null|       S|
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
only showing top 5 rows

Summary of data

titanic_df.describe().show()
+-------+-----------------+-------------------+------------------+--------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+
|summary|      PassengerId|           Survived|            Pclass|                Name|   Sex|               Age|             SibSp|              Parch|            Ticket|             Fare|Cabin|Embarked|
+-------+-----------------+-------------------+------------------+--------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+
|  count|              891|                891|               891|                 891|   891|               714|               891|                891|               891|              891|  204|     889|
|   mean|            446.0| 0.3838383838383838| 2.308641975308642|                null|  null| 29.69911764705882|0.5230078563411896|0.38159371492704824|260318.54916792738| 32.2042079685746| null|    null|
| stddev|257.3538420152301|0.48659245426485753|0.8360712409770491|                null|  null|14.526497332334035|1.1027434322934315| 0.8060572211299488|471609.26868834975|49.69342859718089| null|    null|
|    min|                1|                  0|                 1|"Andersson, Mr. A...|female|              0.42|                 0|                  0|            110152|              0.0|  A10|       C|
|    max|              891|                  1|                 3|van Melkebeke, Mr...|  male|              80.0|                 8|                  6|         WE/P 5735|         512.3292|    T|       S|
+-------+-----------------+-------------------+------------------+--------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+

Checking Schema of the dataset

titanic_df.printSchema()
root
 |-- PassengerId: integer (nullable = true)
 |-- Survived: integer (nullable = true)
 |-- Pclass: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: double (nullable = true)
 |-- SibSp: integer (nullable = true)
 |-- Parch: integer (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: double (nullable = true)
 |-- Cabin: string (nullable = true)
 |-- Embarked: string (nullable = true)

Exploratory data analysis (EDA)
Selecting few features

###Checking survival rate using feature class.

titanic_df.select("Survived","Pclass","Embarked").show()
+--------+------+--------+
|Survived|Pclass|Embarked|
+--------+------+--------+
|       0|     3|       S|
|       1|     1|       C|
|       1|     3|       S|
|       1|     1|       S|
|       0|     3|       S|
|       0|     3|       Q|
|       0|     1|       S|
|       0|     3|       S|
|       1|     3|       S|
|       1|     2|       C|
|       1|     3|       S|
|       1|     1|       S|
|       0|     3|       S|
|       0|     3|       S|
|       0|     3|       S|
|       1|     2|       S|
|       0|     3|       Q|
|       1|     2|       S|
|       0|     3|       S|
|       1|     3|       C|
+--------+------+--------+
only showing top 20 rows

groupBy_output = titanic_df.groupBy("Survived", "Pclass").count()
display(groupBy_output)
0.00
100
200
300
400
2
1
3
0.00
100
200
300
400
2
1
3
TOOLTIP
1
0
Pclass
count
Survived
1
0
Here it can be seen that the Pclass1 people were given priority to pclass3 people, even though
We can clearly see that Passenegers Of Pclass 1 were given a very high priority while rescue. Even though the the number of Passengers in Pclass 3 were a lot higher, still the number of survival from them is very low.

Knowing the number of Passengers Survived ?
titanic_df.groupBy("Survived").count().show()
+--------+-----+
|Survived|count|
+--------+-----+
|       1|  342|
|       0|  549|
+--------+-----+

Out of 891 passengers in dataset, only about 342 survived.

To know the particulars about survivors we have to explore more of the data.
The survival rate can be determined by different features of the dataset such as Sex, Port of Embarcation, Age; few to be mentioned.
###Checking survival rate using feature Sex.

titanic_df.groupBy("Sex","Survived").count().show()
+------+--------+-----+
|   Sex|Survived|count|
+------+--------+-----+
|  male|       0|  468|
|female|       1|  233|
|female|       0|   81|
|  male|       1|  109|
+------+--------+-----+

grp_output = titanic_df.groupBy( "Sex").count()
display(grp_output)
female
male
35%
65%
Sex
female
male
Although the number of males are more than females on ship, the female survivors are twice the number of males saved.
### Checking total number of passengers in each Pclass.

a = titanic_df.groupBy("Pclass").count()
display(a)
0.00
100
200
300
400
500
1
3
2
TOOLTIP
Pclass
count
### checking fare with feature pclass.
b = titanic_df.groupBy("Fare", "Pclass").count()
display(b)
0.00
2.0k
4.0k
6.0k
8.0k
1
3
2
TOOLTIP
Pclass
Fare
Handling Null values
# This function use to print feature with null values and null count 
def null_value_count(df):
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)
# Calling function
null_columns_count_list = null_value_count(titanic_df)
spark.createDataFrame(null_columns_count_list, ['Column_With_Null_Value', 'Null_Values_Count']).show()

+----------------------+-----------------+
|Column_With_Null_Value|Null_Values_Count|
+----------------------+-----------------+
|                   Age|              177|
|                 Cabin|              687|
|              Embarked|                2|
+----------------------+-----------------+

Age feature has 177 null values.

mean_age = titanic_df.select(mean('Age')).collect()[0][0]
print(mean_age)
29.69911764705882
titanic_df.select("Name").show()
+--------------------+
|                Name|
+--------------------+
|Braund, Mr. Owen ...|
|Cumings, Mrs. Joh...|
|Heikkinen, Miss. ...|
|Futrelle, Mrs. Ja...|
|Allen, Mr. Willia...|
|    Moran, Mr. James|
|McCarthy, Mr. Tim...|
|Palsson, Master. ...|
|Johnson, Mrs. Osc...|
|Nasser, Mrs. Nich...|
|Sandstrom, Miss. ...|
|Bonnell, Miss. El...|
|Saundercock, Mr. ...|
|Andersson, Mr. An...|
|Vestrom, Miss. Hu...|
|Hewlett, Mrs. (Ma...|
|Rice, Master. Eugene|
|Williams, Mr. Cha...|
|Vander Planke, Mr...|
|Masselmani, Mrs. ...|
+--------------------+
only showing top 20 rows


Show result
To replace these NaN values, we can assign them the mean age of the dataset.But the problem is, there were many people with many different ages. We just cant assign a 4 year kid with the mean age that is 29 years.
we can check the Name feature. Looking upon the feature, we can see that the names have a salutation like Mr or Mrs. Thus we can assign the mean values of Mr and Mrs to the respective groups
###Using the Regex ""[A-Za-z]+)." we extract the initials from the Name. It looks for strings which lie between A-Z or a-z and followed by a .(dot).

titanic_df = titanic_df.withColumn("Initial",regexp_extract(col("Name"),"([A-Za-z]+)\.",1))


titanic_df.show()
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+-------+
|PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|Initial|
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+-------+
|          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|     Mr|
|          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|    Mrs|
|          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|   Miss|
|          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|    Mrs|
|          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| null|       S|     Mr|
|          6|       0|     3|    Moran, Mr. James|  male|null|    0|    0|          330877| 8.4583| null|       Q|     Mr|
|          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|           17463|51.8625|  E46|       S|     Mr|
|          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1|          349909| 21.075| null|       S| Master|
|          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|          347742|11.1333| null|       S|    Mrs|
|         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|          237736|30.0708| null|       C|    Mrs|
|         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|         PP 9549|   16.7|   G6|       S|   Miss|
|         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|          113783|  26.55| C103|       S|   Miss|
|         13|       0|     3|Saundercock, Mr. ...|  male|20.0|    0|    0|       A/5. 2151|   8.05| null|       S|     Mr|
|         14|       0|     3|Andersson, Mr. An...|  male|39.0|    1|    5|          347082| 31.275| null|       S|     Mr|
|         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0|          350406| 7.8542| null|       S|   Miss|
|         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|          248706|   16.0| null|       S|    Mrs|
|         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1|          382652| 29.125| null|       Q| Master|
|         18|       1|     2|Williams, Mr. Cha...|  male|null|    0|    0|          244373|   13.0| null|       S|     Mr|
|         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|          345763|   18.0| null|       S|    Mrs|
|         20|       1|     3|Masselmani, Mrs. ...|female|null|    0|    0|            2649|  7.225| null|       C|    Mrs|
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+-------+
only showing top 20 rows

titanic_df.select("Initial").distinct().show()

+--------+
| Initial|
+--------+
|     Don|
|    Miss|
|Countess|
|     Col|
|     Rev|
|    Lady|
|  Master|
|     Mme|
|    Capt|
|      Mr|
|      Dr|
|     Mrs|
|     Sir|
|Jonkheer|
|    Mlle|
|   Major|
|      Ms|
+--------+

### There are some misspelled Initials like Mlle or Mme that stand for Miss. I will replace them with Miss and same thing for other values.

titanic_df = titanic_df.replace(['Mlle','Mme', 'Ms', 'Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
               ['Miss','Miss','Miss','Mr','Mr',  'Mrs',  'Mrs',  'Other',  'Other','Other','Mr','Mr','Mr'])

titanic_df.select("Initial").distinct().show()

+-------+
|Initial|
+-------+
|   Miss|
|  Other|
| Master|
|     Mr|
|    Mrs|
+-------+

###lets check the average age by Initials
titanic_df.groupby('Initial').avg('Age').collect()
Out[89]: [Row(Initial='Miss', avg(Age)=21.86),
 Row(Initial='Other', avg(Age)=45.888888888888886),
 Row(Initial='Master', avg(Age)=4.574166666666667),
 Row(Initial='Mr', avg(Age)=32.73960880195599),
 Row(Initial='Mrs', avg(Age)=35.981818181818184)]
###Let's impute missing values in age feature based on average age of Initials

titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Miss") & (titanic_df["Age"].isNull()), 22).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Other") & (titanic_df["Age"].isNull()), 46).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Master") & (titanic_df["Age"].isNull()), 5).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mr") & (titanic_df["Age"].isNull()), 33).otherwise(titanic_df["Age"]))
titanic_df = titanic_df.withColumn("Age",when((titanic_df["Initial"] == "Mrs") & (titanic_df["Age"].isNull()), 36).otherwise(titanic_df["Age"]))

Check the imputation

###Check the imputation

titanic_df.filter(titanic_df.Age==46).select("Initial").show()

+-------+
|Initial|
+-------+
|     Mr|
|     Mr|
|     Mr|
+-------+

titanic_df.select("Age").show()
+----+
| Age|
+----+
|22.0|
|38.0|
|26.0|
|35.0|
|35.0|
|33.0|
|54.0|
| 2.0|
|27.0|
|14.0|
| 4.0|
|58.0|
|20.0|
|39.0|
|14.0|
|55.0|
| 2.0|
|33.0|
|31.0|
|36.0|
+----+
only showing top 20 rows

###Embarked feature has only two missining values. Let's check values within Embarked
titanic_df.groupBy("Embarked").count().show()
+--------+-----+
|Embarked|count|
+--------+-----+
|       Q|   77|
|    null|    2|
|       C|  168|
|       S|  644|
+--------+-----+

###Majority Passengers boarded from "S". We can impute with "S"
titanic_df = titanic_df.na.fill({"Embarked" : 'S'})

###We can drop Cabin features as it has lots of null values
titanic_df = titanic_df.drop("Cabin")
titanic_df.printSchema()
root
 |-- PassengerId: integer (nullable = true)
 |-- Survived: integer (nullable = true)
 |-- Pclass: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: double (nullable = true)
 |-- SibSp: integer (nullable = true)
 |-- Parch: integer (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: double (nullable = true)
 |-- Embarked: string (nullable = false)
 |-- Initial: string (nullable = true)

We can create a new feature called "Family_size" and "Alone" and analyse it. This feature is the summation of Parch(parents/children) and SibSp(siblings/spouses). It gives us a combined data so that we can check if survival rate have anything to do with family size of the passengers

titanic_df = titanic_df.withColumn("Family_Size",col('SibSp')+col('Parch'))
titanic_df.groupBy("Family_Size").count().show()
+-----------+-----+
|Family_Size|count|
+-----------+-----+
|          1|  161|
|          6|   12|
|          3|   29|
|          5|   22|
|          4|   15|
|          7|    6|
|         10|    7|
|          2|  102|
|          0|  537|
+-----------+-----+

titanic_df = titanic_df.withColumn('Alone',lit(0))
titanic_df = titanic_df.withColumn("Alone",when(titanic_df["Family_Size"] == 0, 1).otherwise(titanic_df["Alone"]))

titanic_df.columns
Out[100]: ['PassengerId',
 'Survived',
 'Pclass',
 'Name',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Ticket',
 'Fare',
 'Embarked',
 'Initial',
 'Family_Size',
 'Alone']
###Lets convert Sex, Embarked & Initial columns from string to number using StringIndexer.

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanic_df) for column in ["Sex","Embarked","Initial"]]
pipeline = Pipeline(stages=indexers)
titanic_df = pipeline.fit(titanic_df).transform(titanic_df)
titanic_df.show()
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+--------+-------+-----------+-----+---------+--------------+-------------+
|PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Embarked|Initial|Family_Size|Alone|Sex_index|Embarked_index|Initial_index|
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+--------+-------+-----------+-----+---------+--------------+-------------+
|          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25|       S|     Mr|          1|    0|      0.0|           0.0|          0.0|
|          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|       C|    Mrs|          1|    0|      1.0|           1.0|          2.0|
|          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925|       S|   Miss|          0|    1|      1.0|           0.0|          1.0|
|          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1|       S|    Mrs|          1|    0|      1.0|           0.0|          2.0|
|          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05|       S|     Mr|          0|    1|      0.0|           0.0|          0.0|
|          6|       0|     3|    Moran, Mr. James|  male|33.0|    0|    0|          330877| 8.4583|       Q|     Mr|          0|    1|      0.0|           2.0|          0.0|
|          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|           17463|51.8625|       S|     Mr|          0|    1|      0.0|           0.0|          0.0|
|          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1|          349909| 21.075|       S| Master|          4|    0|      0.0|           0.0|          3.0|
|          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|          347742|11.1333|       S|    Mrs|          2|    0|      1.0|           0.0|          2.0|
|         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|          237736|30.0708|       C|    Mrs|          1|    0|      1.0|           1.0|          2.0|
|         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|         PP 9549|   16.7|       S|   Miss|          2|    0|      1.0|           0.0|          1.0|
|         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|          113783|  26.55|       S|   Miss|          0|    1|      1.0|           0.0|          1.0|
|         13|       0|     3|Saundercock, Mr. ...|  male|20.0|    0|    0|       A/5. 2151|   8.05|       S|     Mr|          0|    1|      0.0|           0.0|          0.0|
|         14|       0|     3|Andersson, Mr. An...|  male|39.0|    1|    5|          347082| 31.275|       S|     Mr|          6|    0|      0.0|           0.0|          0.0|
|         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0|          350406| 7.8542|       S|   Miss|          0|    1|      1.0|           0.0|          1.0|
|         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|          248706|   16.0|       S|    Mrs|          0|    1|      1.0|           0.0|          2.0|
|         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1|          382652| 29.125|       Q| Master|          5|    0|      0.0|           2.0|          3.0|
|         18|       1|     2|Williams, Mr. Cha...|  male|33.0|    0|    0|          244373|   13.0|       S|     Mr|          0|    1|      0.0|           0.0|          0.0|
|         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|          345763|   18.0|       S|    Mrs|          1|    0|      1.0|           0.0|          2.0|
|         20|       1|     3|Masselmani, Mrs. ...|female|36.0|    0|    0|            2649|  7.225|       C|    Mrs|          0|    1|      1.0|           1.0|          2.0|
+-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+--------+-------+-----------+-----+---------+--------------+-------------+
only showing top 20 rows

titanic_df.printSchema()
root
 |-- PassengerId: integer (nullable = true)
 |-- Survived: integer (nullable = true)
 |-- Pclass: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: double (nullable = true)
 |-- SibSp: integer (nullable = true)
 |-- Parch: integer (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: double (nullable = true)
 |-- Embarked: string (nullable = false)
 |-- Initial: string (nullable = true)
 |-- Family_Size: integer (nullable = true)
 |-- Alone: integer (nullable = false)
 |-- Sex_index: double (nullable = false)
 |-- Embarked_index: double (nullable = false)
 |-- Initial_index: double (nullable = false)

###Drop columns which are not required

titanic_df = titanic_df.drop("PassengerId","Name","Ticket","Cabin","Embarked","Sex","Initial")
titanic_df.show()
+--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+-------------+
|Survived|Pclass| Age|SibSp|Parch|   Fare|Family_Size|Alone|Sex_index|Embarked_index|Initial_index|
+--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+-------------+
|       0|     3|22.0|    1|    0|   7.25|          1|    0|      0.0|           0.0|          0.0|
|       1|     1|38.0|    1|    0|71.2833|          1|    0|      1.0|           1.0|          2.0|
|       1|     3|26.0|    0|    0|  7.925|          0|    1|      1.0|           0.0|          1.0|
|       1|     1|35.0|    1|    0|   53.1|          1|    0|      1.0|           0.0|          2.0|
|       0|     3|35.0|    0|    0|   8.05|          0|    1|      0.0|           0.0|          0.0|
|       0|     3|33.0|    0|    0| 8.4583|          0|    1|      0.0|           2.0|          0.0|
|       0|     1|54.0|    0|    0|51.8625|          0|    1|      0.0|           0.0|          0.0|
|       0|     3| 2.0|    3|    1| 21.075|          4|    0|      0.0|           0.0|          3.0|
|       1|     3|27.0|    0|    2|11.1333|          2|    0|      1.0|           0.0|          2.0|
|       1|     2|14.0|    1|    0|30.0708|          1|    0|      1.0|           1.0|          2.0|
|       1|     3| 4.0|    1|    1|   16.7|          2|    0|      1.0|           0.0|          1.0|
|       1|     1|58.0|    0|    0|  26.55|          0|    1|      1.0|           0.0|          1.0|
|       0|     3|20.0|    0|    0|   8.05|          0|    1|      0.0|           0.0|          0.0|
|       0|     3|39.0|    1|    5| 31.275|          6|    0|      0.0|           0.0|          0.0|
|       0|     3|14.0|    0|    0| 7.8542|          0|    1|      1.0|           0.0|          1.0|
|       1|     2|55.0|    0|    0|   16.0|          0|    1|      1.0|           0.0|          2.0|
|       0|     3| 2.0|    4|    1| 29.125|          5|    0|      0.0|           2.0|          3.0|
|       1|     2|33.0|    0|    0|   13.0|          0|    1|      0.0|           0.0|          0.0|
|       0|     3|31.0|    1|    0|   18.0|          1|    0|      1.0|           0.0|          2.0|
|       1|     3|36.0|    0|    0|  7.225|          0|    1|      1.0|           1.0|          2.0|
+--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+-------------+
only showing top 20 rows

Let's put all features into vector
feature = VectorAssembler(inputCols=titanic_df.columns[1:],outputCol="features")
feature_vector= feature.transform(titanic_df)
feature_vector.show()
+--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+-------------+--------------------+
|Survived|Pclass| Age|SibSp|Parch|   Fare|Family_Size|Alone|Sex_index|Embarked_index|Initial_index|            features|
+--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+-------------+--------------------+
|       0|     3|22.0|    1|    0|   7.25|          1|    0|      0.0|           0.0|          0.0|(10,[0,1,2,4,5],[...|
|       1|     1|38.0|    1|    0|71.2833|          1|    0|      1.0|           1.0|          2.0|[1.0,38.0,1.0,0.0...|
|       1|     3|26.0|    0|    0|  7.925|          0|    1|      1.0|           0.0|          1.0|[3.0,26.0,0.0,0.0...|
|       1|     1|35.0|    1|    0|   53.1|          1|    0|      1.0|           0.0|          2.0|[1.0,35.0,1.0,0.0...|
|       0|     3|35.0|    0|    0|   8.05|          0|    1|      0.0|           0.0|          0.0|(10,[0,1,4,6],[3....|
|       0|     3|33.0|    0|    0| 8.4583|          0|    1|      0.0|           2.0|          0.0|(10,[0,1,4,6,8],[...|
|       0|     1|54.0|    0|    0|51.8625|          0|    1|      0.0|           0.0|          0.0|(10,[0,1,4,6],[1....|
|       0|     3| 2.0|    3|    1| 21.075|          4|    0|      0.0|           0.0|          3.0|[3.0,2.0,3.0,1.0,...|
|       1|     3|27.0|    0|    2|11.1333|          2|    0|      1.0|           0.0|          2.0|[3.0,27.0,0.0,2.0...|
|       1|     2|14.0|    1|    0|30.0708|          1|    0|      1.0|           1.0|          2.0|[2.0,14.0,1.0,0.0...|
|       1|     3| 4.0|    1|    1|   16.7|          2|    0|      1.0|           0.0|          1.0|[3.0,4.0,1.0,1.0,...|
|       1|     1|58.0|    0|    0|  26.55|          0|    1|      1.0|           0.0|          1.0|[1.0,58.0,0.0,0.0...|
|       0|     3|20.0|    0|    0|   8.05|          0|    1|      0.0|           0.0|          0.0|(10,[0,1,4,6],[3....|
|       0|     3|39.0|    1|    5| 31.275|          6|    0|      0.0|           0.0|          0.0|[3.0,39.0,1.0,5.0...|
|       0|     3|14.0|    0|    0| 7.8542|          0|    1|      1.0|           0.0|          1.0|[3.0,14.0,0.0,0.0...|
|       1|     2|55.0|    0|    0|   16.0|          0|    1|      1.0|           0.0|          2.0|[2.0,55.0,0.0,0.0...|
|       0|     3| 2.0|    4|    1| 29.125|          5|    0|      0.0|           2.0|          3.0|[3.0,2.0,4.0,1.0,...|
|       1|     2|33.0|    0|    0|   13.0|          0|    1|      0.0|           0.0|          0.0|(10,[0,1,4,6],[2....|
|       0|     3|31.0|    1|    0|   18.0|          1|    0|      1.0|           0.0|          2.0|[3.0,31.0,1.0,0.0...|
|       1|     3|36.0|    0|    0|  7.225|          0|    1|      1.0|           1.0|          2.0|[3.0,36.0,0.0,0.0...|
+--------+------+----+-----+-----+-------+-----------+-----+---------+--------------+-------------+--------------------+
only showing top 20 rows

###Now that the data is all set, let's split it into training and test. I'll be using 80% of it.

(trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)
Modelling
Classification Algorithms used to model the dataset are shown below
LogisticRegression

DecisionTreeClassifier

RandomForestClassifier

Gradient-boosted tree classifier

NaiveBayes

Support Vector Machine

LogisticRegression
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="Survived", featuresCol="features")
#Training algo
lrModel = lr.fit(trainingData)
lr_prediction = lrModel.transform(testData)
lr_prediction.select("prediction", "Survived", "features").show()
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
+----------+--------+--------------------+
|prediction|Survived|            features|
+----------+--------+--------------------+
|       0.0|       0|[1.0,19.0,3.0,2.0...|
|       1.0|       0|[1.0,27.0,0.0,2.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       1.0|       0|[1.0,28.0,1.0,0.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       1.0|       0|(10,[0,1,3,4,5],[...|
|       0.0|       0|(10,[0,1,6],[1.0,...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|[1.0,51.0,0.0,1.0...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
+----------+--------+--------------------+
only showing top 20 rows

Evaluating accuracy of LogisticRegression.
lr_accuracy = evaluator.evaluate(lr_prediction)
print("Accuracy of LogisticRegression is = %g"% (lr_accuracy))
print("Test Error of LogisticRegression = %g " % (1.0 - lr_accuracy))
Accuracy of LogisticRegression is = 0.836257
Test Error of LogisticRegression = 0.163743 
DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
dt_model = dt.fit(trainingData)
dt_prediction = dt_model.transform(testData)
dt_prediction.select("prediction", "Survived", "features").show()

+----------+--------+--------------------+
|prediction|Survived|            features|
+----------+--------+--------------------+
|       0.0|       0|[1.0,19.0,3.0,2.0...|
|       0.0|       0|[1.0,27.0,0.0,2.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|[1.0,28.0,1.0,0.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,3,4,5],[...|
|       0.0|       0|(10,[0,1,6],[1.0,...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|[1.0,51.0,0.0,1.0...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       1.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
+----------+--------+--------------------+
only showing top 20 rows

Evaluating accuracy of DecisionTreeClassifier.
dt_accuracy = evaluator.evaluate(dt_prediction)
print("Accuracy of DecisionTreeClassifier is = %g"% (dt_accuracy))
print("Test Error of DecisionTreeClassifier = %g " % (1.0 - dt_accuracy))

Accuracy of DecisionTreeClassifier is = 0.807018
Test Error of DecisionTreeClassifier = 0.192982 
RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier
rf = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
rf_model = rf.fit(trainingData)
rf_prediction = rf_model.transform(testData)
rf_prediction.select("prediction", "Survived", "features").show()
+----------+--------+--------------------+
|prediction|Survived|            features|
+----------+--------+--------------------+
|       0.0|       0|[1.0,19.0,3.0,2.0...|
|       0.0|       0|[1.0,27.0,0.0,2.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|[1.0,28.0,1.0,0.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,3,4,5],[...|
|       0.0|       0|(10,[0,1,6],[1.0,...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|[1.0,51.0,0.0,1.0...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       1.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
+----------+--------+--------------------+
only showing top 20 rows

Evaluating accuracy of RandomForestClassifier.
rf_accuracy = evaluator.evaluate(rf_prediction)
print("Accuracy of RandomForestClassifier is = %g"% (rf_accuracy))
print("Test Error of RandomForestClassifier  = %g " % (1.0 - rf_accuracy))
Accuracy of RandomForestClassifier is = 0.807018
Test Error of RandomForestClassifier  = 0.192982 
Gradient-boosted tree classifier
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="Survived", featuresCol="features",maxIter=10)
gbt_model = gbt.fit(trainingData)
gbt_prediction = gbt_model.transform(testData)
gbt_prediction.select("prediction", "Survived", "features").show()

+----------+--------+--------------------+
|prediction|Survived|            features|
+----------+--------+--------------------+
|       0.0|       0|[1.0,19.0,3.0,2.0...|
|       1.0|       0|[1.0,27.0,0.0,2.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       1.0|       0|[1.0,28.0,1.0,0.0...|
|       1.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,3,4,5],[...|
|       0.0|       0|(10,[0,1,6],[1.0,...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|[1.0,51.0,0.0,1.0...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
+----------+--------+--------------------+
only showing top 20 rows

Evaluate accuracy of Gradient-boosted.
gbt_accuracy = evaluator.evaluate(gbt_prediction)
print("Accuracy of Gradient-boosted tree classifie is = %g"% (gbt_accuracy))
print("Test Error of Gradient-boosted tree classifie %g"% (1.0 - gbt_accuracy))
Accuracy of Gradient-boosted tree classifie is = 0.824561
Test Error of Gradient-boosted tree classifie 0.175439
NaiveBayes
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(labelCol="Survived", featuresCol="features")
nb_model = nb.fit(trainingData)
nb_prediction = nb_model.transform(testData)
nb_prediction.select("prediction", "Survived", "features").show()

+----------+--------+--------------------+
|prediction|Survived|            features|
+----------+--------+--------------------+
|       1.0|       0|[1.0,19.0,3.0,2.0...|
|       1.0|       0|[1.0,27.0,0.0,2.0...|
|       1.0|       0|(10,[0,1,4,6],[1....|
|       1.0|       0|[1.0,28.0,1.0,0.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       1.0|       0|(10,[0,1,3,4,5],[...|
|       0.0|       0|(10,[0,1,6],[1.0,...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       1.0|       0|(10,[0,1,2,4,5],[...|
|       1.0|       0|[1.0,51.0,0.0,1.0...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       1.0|       0|(10,[0,1,4,6],[2....|
|       1.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       1.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
+----------+--------+--------------------+
only showing top 20 rows

Evaluating accuracy of NaiveBayes.
nb_accuracy = evaluator.evaluate(nb_prediction)
print("Accuracy of NaiveBayes is  = %g"% (nb_accuracy))
print("Test Error of NaiveBayes  = %g " % (1.0 - nb_accuracy))
Accuracy of NaiveBayes is  = 0.695906
Test Error of NaiveBayes  = 0.304094 
Support Vector Machine
from pyspark.ml.classification import LinearSVC
svm = LinearSVC(labelCol="Survived", featuresCol="features")
svm_model = svm.fit(trainingData)
svm_prediction = svm_model.transform(testData)
svm_prediction.select("prediction", "Survived", "features").show()

+----------+--------+--------------------+
|prediction|Survived|            features|
+----------+--------+--------------------+
|       0.0|       0|[1.0,19.0,3.0,2.0...|
|       0.0|       0|[1.0,27.0,0.0,2.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|[1.0,28.0,1.0,0.0...|
|       0.0|       0|(10,[0,1,4,6],[1....|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,3,4,5],[...|
|       0.0|       0|(10,[0,1,6],[1.0,...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|[1.0,51.0,0.0,1.0...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6,8],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,4,6],[2....|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,2,4,5],[...|
|       0.0|       0|(10,[0,1,4,6],[2....|
+----------+--------+--------------------+
only showing top 20 rows

Evaluating the accuracy of Support Vector Machine.
svm_accuracy = evaluator.evaluate(svm_prediction)
print("Accuracy of Support Vector Machine is = %g"% (svm_accuracy))
print("Test Error of Support Vector Machine = %g " % (1.0 - svm_accuracy))
Accuracy of Support Vector Machine is = 0.836257
Test Error of Support Vector Machine = 0.163743 
