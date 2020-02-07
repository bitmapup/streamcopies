# streamcopies

## Bibliography

* Craven, M., & Shavlik, J. W. (1996). Extracting tree-structured representations of trained networks. In Advances in neural information processing systems (pp. 24-30).
* Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
* Buciluǎ, C., Caruana, R., & Niculescu-Mizil, A. (2006, August). Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 535-541).

* Domingos, P., & Hulten, G. (2000, August). Mining high-speed data streams. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 71-80).
* Montiel López, J. (2019). Fast and slow machine learning (Doctoral dissertation, Paris Saclay).


## Data dictionary

	* client_id,    #-- 0  
        * social_class ,     #-- 1
        * date,              #-- 2
        * mcc,               #-- 3
        * country_code,      #-- 4
        * amount_usd,        #-- 5 *
        * client_age,        #-- 6 *
        * client_gender,     #-- 7
        * debit_type,        #-- 8 Credit or debit card
        * agency_id,         #-- 9
        * agency_departement,#-- 10 
        * agency_province,   #-- 11
        * agency_district,   #-- 12
        * agency_lima,       #-- 13
        * agency_region,     #-- 14
        * merchant_id,       #-- 15
        * merchant_departement, #-- 16
        * merchant_province, #-- 17
        * merchant_district, #-- 18
        * merchant_lon,      #-- 19 *
        * merchant_lat       #-- 20*

\* Numeric variable, all other are categorical
