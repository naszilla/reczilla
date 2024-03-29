# this is the list of subdirectories in gs://reczilla-results/dataset-splits/splits-v5
v5_dataset_list=(
AmazonAllBeauty
AmazonAllElectronics
AmazonAlternativeRock
AmazonAmazonFashion
AmazonAmazonInstantVideo
AmazonAppliances
AmazonAppsforAndroid
AmazonAppstoreforAndroid
AmazonArtsCraftsSewing
AmazonAutomotive
AmazonBaby
AmazonBabyProducts
AmazonBeauty
AmazonBlues
AmazonBooks
AmazonBuyaKindle
AmazonCDsVinyl
AmazonCellPhonesAccessories
AmazonChristian
AmazonClassical
AmazonClothingShoesJewelry
AmazonCollectiblesFineArt
AmazonComputers
AmazonCountry
AmazonDanceElectronic
AmazonDavis
AmazonDigitalMusic
AmazonElectronics
AmazonFolk
AmazonGiftCards
AmazonGospel
AmazonGroceryGourmetFood
AmazonHardRockMetal
AmazonHealthPersonalCare
AmazonHomeImprovement
AmazonHomeKitchen
AmazonIndustrialScientific
AmazonInternational
AmazonJazz
AmazonKindleStore
AmazonKitchenDining
AmazonLatinMusic
AmazonLuxuryBeauty
AmazonMP3PlayersAccessories
AmazonMagazineSubscriptions
AmazonMiscellaneous
AmazonMoviesTV
AmazonMusicalInstruments
AmazonNewAge
AmazonOfficeProducts
AmazonOfficeSchoolSupplies
AmazonPatioLawnGarden
AmazonPetSupplies
AmazonPop
AmazonPurchaseCircles
AmazonRB
AmazonRapHipHop
AmazonRock
AmazonSoftware
AmazonSportsOutdoors
AmazonToolsHomeImprovement
AmazonToysGames
AmazonVideoGames
AmazonWine
Anime
BookCrossing
CiaoDVD
Dating
Epinions
FilmTrust
Frappe
GoogleLocalReviews
Gowalla
Jester2
LastFM
MarketBiasAmazon
MarketBiasModCloth
MovieTweetings
Movielens100K
Movielens10M
Movielens1M
Movielens20M
MovielensHetrec2011
NetflixPrize
Recipes
Wikilens
YahooMovies
YahooMusic
)

# remove all amazon except moviestv and homeimprovement, and movielens 1m and 20m (keep 100k and 10m)
v5_dataset_list_small=(
AmazonMoviesTV
AmazonHomeImprovement
Anime
BookCrossing
CiaoDVD
Dating
Epinions
FilmTrust
Frappe
GoogleLocalReviews
Gowalla
Jester2
LastFM
MarketBiasAmazon
MarketBiasModCloth
MovieTweetings
Movielens100K
Movielens10M
MovielensHetrec2011
NetflixPrize
Recipes
Wikilens
YahooMovies
YahooMusic
)

# this is the remaining datasets, not included in _small (above)
v5_dataset_list_holdout=(
Movielens1M
Movielens20M
AmazonAllBeauty
AmazonAllElectronics
AmazonAlternativeRock
AmazonAmazonFashion
AmazonAmazonInstantVideo
AmazonAppliances
AmazonAppsforAndroid
AmazonAppstoreforAndroid
AmazonArtsCraftsSewing
AmazonAutomotive
AmazonBaby
AmazonBabyProducts
AmazonBeauty
AmazonBlues
AmazonBooks
AmazonBuyaKindle
AmazonCDsVinyl
AmazonCellPhonesAccessories
AmazonChristian
AmazonClassical
AmazonClothingShoesJewelry
AmazonCollectiblesFineArt
AmazonComputers
AmazonCountry
AmazonDanceElectronic
AmazonDavis
AmazonDigitalMusic
AmazonElectronics
AmazonFolk
AmazonGiftCards
AmazonGospel
AmazonGroceryGourmetFood
AmazonHardRockMetal
AmazonHealthPersonalCare
AmazonHomeKitchen
AmazonIndustrialScientific
AmazonInternational
AmazonJazz
AmazonKindleStore
AmazonKitchenDining
AmazonLatinMusic
AmazonLuxuryBeauty
AmazonMP3PlayersAccessories
AmazonMagazineSubscriptions
AmazonMiscellaneous
AmazonMusicalInstruments
AmazonNewAge
AmazonOfficeProducts
AmazonOfficeSchoolSupplies
AmazonPatioLawnGarden
AmazonPetSupplies
AmazonPop
AmazonPurchaseCircles
AmazonRB
AmazonRapHipHop
AmazonRock
AmazonSoftware
AmazonSportsOutdoors
AmazonToolsHomeImprovement
AmazonToysGames
AmazonVideoGames
AmazonWine
)