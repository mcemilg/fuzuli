# Fuzuli

Fuzuli is a basic implementation of [encoder-decoder](https://github.com/tensorflow/nmt) for nmt. Old Turkish poems have certain styles by line.
And the styles are like a basic language. 
This model implemented for translate these poems to their equivalent style. 

*_Türkçe döküman [buradan](README.tr.md)._*

Here is an example:
```
Poem
Eksik olmaz gamumuz bunca ki bizden gam alub
Her gelen gamlı gider şâd gelüb yanumuza

Style
Feilâtün Fâilâtün Feilâtün Feilâtün Feilün Fa'lün
```

Check [here](http://www.wiki-zero.com/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvQXJhYmljX3Byb3NvZHk) for more information about this poetic meter.

**_Note_**: This was an exprimental project.


### Data
```
~700 Poem
~7000 line
58 style
```

### Training
Usage:
```
python nmt.py ./../data/
```

**Note**: You can stop at any point because it creates checkpoint on each test.

Sample:
```
Loss :  151.909
---Test---
 -Predictions- 
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffFt</go>tnnlututututututuMMMMMMMM'a11111nnenlelelue
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffFt</go>tnnlututututututuMMMMMMMM'a11111nnenlelelue
MMffFt</go>tnnlututututututuMMMMMMMM'a11111nnenlelelue
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
 -Targets- 
Mef'ûlü Fâilâtü Mefâîlü Fâilün</go>                   
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Müstef'ilün Müstef'ilün</go>                          
Mefâîlün Mefâîlün Mefâîlün Mefâîlün</go>              
Fâ'ilâtün Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>            
Fâ'ilâtün Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>            
Müfte'îlün Mefâ'îlün Müfte'îlün Mefâ'îlün</go>        
Müfte'îlün Mefâ'îlün Müfte'îlün Mefâ'îlün</go>        
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Fâ'ilâtün Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>            
Mefâ'îlün Mefâ'îlün Fe'ûlün</go>                      
Müfte'îlün Mefâ'îlün Müfte'îlün Mefâ'îlün</go>        
Mef'ûlü Fâ'ilâtü Mefâ'îlü Fâ'ilün</go>                
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Feilâtün Fâilâtün Feilâtün Feilâtün Feilün Fa'lün</go>
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Mefâ'îlün Mefâ'îlün Mefâ'îlün Mefâ'îlün</go>          
Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>                      
Mef'ûlü Fâilâtü Mefâîlü Fâilün</go> 
```

### Demo
Demo works with saved checkpoint. Do not remember to change checkpoint name at demo.

Usage
```
python nmt_demo.py ./../data/
```

### Versions
```
tensorflow                1.4.1
numpy                     1.11.3
pandas                    0.22.0
```


