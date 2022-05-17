family_map = {
    "Amazon": {
        "AmazonAllBeautyReader",
        "AmazonAllElectronicsReader",
        "AmazonAlternativeRockReader",
        "AmazonAmazonFashionReader",
        "AmazonAmazonInstantVideoReader",
        "AmazonAppliancesReader",
        "AmazonAppsforAndroidReader",
        "AmazonAppstoreforAndroidReader",
        "AmazonArtsCraftsSewingReader",
        "AmazonAutomotiveReader",
        "AmazonBabyReader",
        "AmazonBabyProductsReader",
        "AmazonBeautyReader",
        "AmazonBluesReader",
        "AmazonBooksReader",
        "AmazonBuyaKindleReader",
        "AmazonCDsVinylReader",
        "AmazonCellPhonesAccessoriesReader",
        "AmazonChristianReader",
        "AmazonClassicalReader",
        "AmazonClothingShoesJewelryReader",
        "AmazonCollectiblesFineArtReader",
        "AmazonComputersReader",
        "AmazonCountryReader",
        "AmazonDanceElectronicReader",
        "AmazonDavisReader",
        "AmazonDigitalMusicReader",
        "AmazonElectronicsReader",
        "AmazonFolkReader",
        "AmazonGiftCardsReader",
        "AmazonGospelReader",
        "AmazonGroceryGourmetFoodReader",
        "AmazonHardRockMetalReader",
        "AmazonHealthPersonalCareReader",
        "AmazonHomeImprovementReader",
        "AmazonHomeKitchenReader",
        "AmazonIndustrialScientificReader",
        "AmazonInternationalReader",
        "AmazonJazzReader",
        "AmazonKindleStoreReader",
        "AmazonKitchenDiningReader",
        "AmazonLatinMusicReader",
        "AmazonLuxuryBeautyReader",
        "AmazonMagazineSubscriptionsReader",
        "AmazonMiscellaneousReader",
        "AmazonMoviesTVReader",
        "AmazonMP3PlayersAccessoriesReader",
        "AmazonMusicalInstrumentsReader",
        "AmazonNewAgeReader",
        "AmazonOfficeProductsReader",
        "AmazonOfficeSchoolSuppliesReader",
        "AmazonPatioLawnGardenReader",
        "AmazonPetSuppliesReader",
        "AmazonPopReader",
        "AmazonPurchaseCirclesReader",
        "AmazonRapHipHopReader",
        "AmazonRBReader",
        "AmazonRockReader",
        "AmazonSoftwareReader",
        "AmazonSportsOutdoorsReader",
        "AmazonToolsHomeImprovementReader",
        "AmazonToysGamesReader",
        "AmazonVideoGamesReader",
        "AmazonWineReader"
    },
    "Movielens": {
        "Movielens100KReader",
        "Movielens10MReader",
        "Movielens1MReader",
        "Movielens20MReader",
        "MovielensHetrec2011Reader"
    },
    "Yahoo": {
        "YahooMoviesReader",
        "YahooMusicReader"
    }
}

unique_datasets = {
    "AnimeReader",
    "BookCrossingReader",
    "CiaoDVDReader",
    "DatingReader",
    "EpinionsReader",
    "FilmTrustReader",
    "FrappeReader",
    "GoogleLocalReviewsReader",
    "GowallaReader",
    "Jester2Reader",
    "LastFMReader",
    "MarketBiasAmazonReader",
    "MarketBiasModClothReader",
    "MovieTweetingsReader",
    "NetflixPrizeReader",
    "RecipesReader",
    "WikilensReader"
}

reverse_family_map = {
    dataset: family for family, datasets in family_map.items()
    for dataset in datasets
}

reverse_family_map.update({
    dataset: dataset for dataset in unique_datasets
})

def dataset_family_lookup(dataset_name, strict_match=True):
    """Looks up the dataset family corresponding to the dataset."""
    if strict_match:
        if dataset_name not in reverse_family_map:
            raise RuntimeError(f"Strict match for dataset name not found: {dataset_name}")
        return reverse_family_map[dataset_name]
    else:
        return reverse_family_map.get(dataset_name, dataset_name)

def get_dataset_families():
    """This returns all of the dataset families (useful for hold-one-out validation with dataset families)"""
    return set(family_map.keys()).union(unique_datasets)

def get_all_datasets():
    """Returns all of the datasets we have."""
    return set([k for v in family_map.values() for k in v]).union(unique_datasets)