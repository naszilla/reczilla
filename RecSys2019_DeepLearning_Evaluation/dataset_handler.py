#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 4 Feb 2022

@author: yyyyyy C yyyyyy

code for iterating over each dataset


"""

# multiple amazon reviewdata readers
from Data_manager.AmazonReviewData.AmazonAllBeautyReader import AmazonAllBeautyReader
from Data_manager.AmazonReviewData.AmazonAllCreditCardsReader import AmazonAllCreditCardsReader
from Data_manager.AmazonReviewData.AmazonAllElectronicsReader import AmazonAllElectronicsReader
from Data_manager.AmazonReviewData.AmazonAlternativeRockReader import AmazonAlternativeRockReader
from Data_manager.AmazonReviewData.AmazonAmazonCoinsReader import AmazonAmazonCoinsReader
from Data_manager.AmazonReviewData.AmazonAmazonFashionReader import AmazonAmazonFashionReader
from Data_manager.AmazonReviewData.AmazonAmazonFireTVReader import AmazonAmazonFireTVReader
from Data_manager.AmazonReviewData.AmazonAmazonInstantVideoReader import AmazonAmazonInstantVideoReader
from Data_manager.AmazonReviewData.AmazonAppliancesReader import AmazonAppliancesReader
from Data_manager.AmazonReviewData.AmazonAppsforAndroidReader import AmazonAppsforAndroidReader
from Data_manager.AmazonReviewData.AmazonAppstoreforAndroidReader import AmazonAppstoreforAndroidReader
from Data_manager.AmazonReviewData.AmazonArtsCraftsSewingReader import AmazonArtsCraftsSewingReader
from Data_manager.AmazonReviewData.AmazonAutomotiveReader import AmazonAutomotiveReader
from Data_manager.AmazonReviewData.AmazonBabyReader import AmazonBabyReader
from Data_manager.AmazonReviewData.AmazonBabyProductsReader import AmazonBabyProductsReader
from Data_manager.AmazonReviewData.AmazonBeautyReader import AmazonBeautyReader
from Data_manager.AmazonReviewData.AmazonBluesReader import AmazonBluesReader
from Data_manager.AmazonReviewData.AmazonBooksReader import AmazonBooksReader
from Data_manager.AmazonReviewData.AmazonBroadwayVocalistsReader import AmazonBroadwayVocalistsReader
from Data_manager.AmazonReviewData.AmazonBuyaKindleReader import AmazonBuyaKindleReader
from Data_manager.AmazonReviewData.AmazonCDsVinylReader import AmazonCDsVinylReader
from Data_manager.AmazonReviewData.AmazonCameraPhotoReader import AmazonCameraPhotoReader
from Data_manager.AmazonReviewData.AmazonCarElectronicsReader import AmazonCarElectronicsReader
from Data_manager.AmazonReviewData.AmazonCelebrateyourBirthdaywithNickelodeonReader import AmazonCelebrateyourBirthdaywithNickelodeonReader
from Data_manager.AmazonReviewData.AmazonCellPhonesAccessoriesReader import AmazonCellPhonesAccessoriesReader
from Data_manager.AmazonReviewData.AmazonChildrensMusicReader import AmazonChildrensMusicReader
from Data_manager.AmazonReviewData.AmazonChristianReader import AmazonChristianReader
from Data_manager.AmazonReviewData.AmazonClassicRockReader import AmazonClassicRockReader
from Data_manager.AmazonReviewData.AmazonClassicalReader import AmazonClassicalReader
from Data_manager.AmazonReviewData.AmazonClothingShoesJewelryReader import AmazonClothingShoesJewelryReader
from Data_manager.AmazonReviewData.AmazonCollectibleCoinsReader import AmazonCollectibleCoinsReader
from Data_manager.AmazonReviewData.AmazonCollectiblesFineArtReader import AmazonCollectiblesFineArtReader
from Data_manager.AmazonReviewData.AmazonComputersReader import AmazonComputersReader
from Data_manager.AmazonReviewData.AmazonCountryReader import AmazonCountryReader
from Data_manager.AmazonReviewData.AmazonDanceElectronicReader import AmazonDanceElectronicReader
from Data_manager.AmazonReviewData.AmazonDavisReader import AmazonDavisReader
from Data_manager.AmazonReviewData.AmazonDigitalMusicReader import AmazonDigitalMusicReader
from Data_manager.AmazonReviewData.AmazonElectronicsReader import AmazonElectronicsReader
from Data_manager.AmazonReviewData.AmazonEntertainmentReader import AmazonEntertainmentReader
from Data_manager.AmazonReviewData.AmazonFolkReader import AmazonFolkReader
from Data_manager.AmazonReviewData.AmazonFurnitureDecorReader import AmazonFurnitureDecorReader
from Data_manager.AmazonReviewData.AmazonGPSNavigationReader import AmazonGPSNavigationReader
from Data_manager.AmazonReviewData.AmazonGiftCardsReader import AmazonGiftCardsReader
from Data_manager.AmazonReviewData.AmazonGiftCardsStoreReader import AmazonGiftCardsStoreReader
from Data_manager.AmazonReviewData.AmazonGospelReader import AmazonGospelReader
from Data_manager.AmazonReviewData.AmazonGroceryGourmetFoodReader import AmazonGroceryGourmetFoodReader
from Data_manager.AmazonReviewData.AmazonHardRockMetalReader import AmazonHardRockMetalReader
from Data_manager.AmazonReviewData.AmazonHealthPersonalCareReader import AmazonHealthPersonalCareReader
from Data_manager.AmazonReviewData.AmazonHomeImprovementReader import AmazonHomeImprovementReader
from Data_manager.AmazonReviewData.AmazonHomeKitchenReader import AmazonHomeKitchenReader
from Data_manager.AmazonReviewData.AmazonIndustrialScientificReader import AmazonIndustrialScientificReader
from Data_manager.AmazonReviewData.AmazonInternationalReader import AmazonInternationalReader
from Data_manager.AmazonReviewData.AmazonJazzReader import AmazonJazzReader
from Data_manager.AmazonReviewData.AmazonKindleStoreReader import AmazonKindleStoreReader
from Data_manager.AmazonReviewData.AmazonKitchenDiningReader import AmazonKitchenDiningReader
from Data_manager.AmazonReviewData.AmazonLatinMusicReader import AmazonLatinMusicReader
from Data_manager.AmazonReviewData.AmazonLearningEducationReader import AmazonLearningEducationReader
from Data_manager.AmazonReviewData.AmazonLuxuryBeautyReader import AmazonLuxuryBeautyReader
from Data_manager.AmazonReviewData.AmazonMP3PlayersAccessoriesReader import AmazonMP3PlayersAccessoriesReader
from Data_manager.AmazonReviewData.AmazonMagazineSubscriptionsReader import AmazonMagazineSubscriptionsReader
from Data_manager.AmazonReviewData.AmazonMicrosoftReader import AmazonMicrosoftReader
from Data_manager.AmazonReviewData.AmazonMiscellaneousReader import AmazonMiscellaneousReader
from Data_manager.AmazonReviewData.AmazonMoviesTVReader import AmazonMoviesTVReader
from Data_manager.AmazonReviewData.AmazonMusicalInstrumentsReader import AmazonMusicalInstrumentsReader
from Data_manager.AmazonReviewData.AmazonNewAgeReader import AmazonNewAgeReader
from Data_manager.AmazonReviewData.AmazonNickelodeonReader import AmazonNickelodeonReader
from Data_manager.AmazonReviewData.AmazonOfficeProductsReader import AmazonOfficeProductsReader
from Data_manager.AmazonReviewData.AmazonOfficeSchoolSuppliesReader import AmazonOfficeSchoolSuppliesReader
from Data_manager.AmazonReviewData.AmazonPatioLawnGardenReader import AmazonPatioLawnGardenReader
from Data_manager.AmazonReviewData.AmazonPetSuppliesReader import AmazonPetSuppliesReader
from Data_manager.AmazonReviewData.AmazonPopReader import AmazonPopReader
from Data_manager.AmazonReviewData.AmazonPublishersReader import AmazonPublishersReader
from Data_manager.AmazonReviewData.AmazonPurchaseCirclesReader import AmazonPurchaseCirclesReader
from Data_manager.AmazonReviewData.AmazonRBReader import AmazonRBReader
from Data_manager.AmazonReviewData.AmazonRapHipHopReader import AmazonRapHipHopReader
from Data_manager.AmazonReviewData.AmazonRockReader import AmazonRockReader
from Data_manager.AmazonReviewData.AmazonSoftwareReader import AmazonSoftwareReader
from Data_manager.AmazonReviewData.AmazonSportsCollectiblesReader import AmazonSportsCollectiblesReader
from Data_manager.AmazonReviewData.AmazonSportsOutdoorsReader import AmazonSportsOutdoorsReader
from Data_manager.AmazonReviewData.AmazonToolsHomeImprovementReader import AmazonToolsHomeImprovementReader
from Data_manager.AmazonReviewData.AmazonToysGamesReader import AmazonToysGamesReader
from Data_manager.AmazonReviewData.AmazonVideoGamesReader import AmazonVideoGamesReader
from Data_manager.AmazonReviewData.AmazonWineReader import AmazonWineReader

# multiple movielens readers
from Data_manager.Movielens.Movielens100KReader import Movielens100KReader
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader

# everything else!
from Data_manager.BookCrossing.BookCrossingReader import BookCrossingReader
from Data_manager.Dating.DatingReader import DatingReader
from Data_manager.Epinions.EpinionsReader import EpinionsReader
from Data_manager.FilmTrust.FilmTrustReader import FilmTrustReader
from Data_manager.Frappe.FrappeReader import FrappeReader
from Data_manager.Gowalla.GowallaReader import GowallaReader
from Data_manager.Jester2.Jester2Reader import Jester2Reader
from Data_manager.MarketBiasAmazon.MarketBiasAmazonReader import MarketBiasAmazonReader
from Data_manager.MarketBiasModCloth.MarketBiasModClothReader import (
    MarketBiasModClothReader,
)
from Data_manager.MovieTweetings.MovieTweetingsReader import MovieTweetingsReader
from Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from Data_manager.Recipes.RecipesReader import RecipesReader
from Data_manager.Wikilens.WikilensReader import WikilensReader
from Data_manager.Anime.AnimeReader import AnimeReader
from Data_manager.CiaoDVD.CiaoDVDReader import CiaoDVDReader
from Data_manager.GoogleLocalReviews.GoogleLocalReviewsReader import GoogleLocalReviewsReader
from Data_manager.LastFM.LastFMReader import LastFMReader

DATASET_READER_LIST = [
    AmazonAllBeautyReader,
    AmazonAllCreditCardsReader,
    AmazonAllElectronicsReader,
    AmazonAlternativeRockReader,
    AmazonAmazonCoinsReader,
    AmazonAmazonFashionReader,
    AmazonAmazonFireTVReader,
    AmazonAmazonInstantVideoReader,
    AmazonAppliancesReader,
    AmazonAppsforAndroidReader,
    AmazonAppstoreforAndroidReader,
    AmazonArtsCraftsSewingReader,
    AmazonAutomotiveReader,
    AmazonBabyReader,
    AmazonBabyProductsReader,
    AmazonBeautyReader,
    AmazonBluesReader,
    AmazonBooksReader,
    AmazonBroadwayVocalistsReader,
    AmazonBuyaKindleReader,
    AmazonCDsVinylReader,
    AmazonCameraPhotoReader,
    AmazonCarElectronicsReader,
    AmazonCelebrateyourBirthdaywithNickelodeonReader,
    AmazonCellPhonesAccessoriesReader,
    AmazonChildrensMusicReader,
    AmazonChristianReader,
    AmazonClassicRockReader,
    AmazonClassicalReader,
    AmazonClothingShoesJewelryReader,
    AmazonCollectibleCoinsReader,
    AmazonCollectiblesFineArtReader,
    AmazonComputersReader,
    AmazonCountryReader,
    AmazonDanceElectronicReader,
    AmazonDavisReader,
    AmazonDigitalMusicReader,
    AmazonElectronicsReader,
    AmazonEntertainmentReader,
    AmazonFolkReader,
    AmazonFurnitureDecorReader,
    AmazonGPSNavigationReader,
    AmazonGiftCardsReader,
    AmazonGiftCardsStoreReader,
    AmazonGospelReader,
    AmazonGroceryGourmetFoodReader,
    AmazonHardRockMetalReader,
    AmazonHealthPersonalCareReader,
    AmazonHomeImprovementReader,
    AmazonHomeKitchenReader,
    AmazonIndustrialScientificReader,
    AmazonInternationalReader,
    AmazonJazzReader,
    AmazonKindleStoreReader,
    AmazonKitchenDiningReader,
    AmazonLatinMusicReader,
    AmazonLearningEducationReader,
    AmazonLuxuryBeautyReader,
    AmazonMP3PlayersAccessoriesReader,
    AmazonMagazineSubscriptionsReader,
    AmazonMicrosoftReader,
    AmazonMiscellaneousReader,
    AmazonMoviesTVReader,
    AmazonMusicalInstrumentsReader,
    AmazonNewAgeReader,
    AmazonNickelodeonReader,
    AmazonOfficeProductsReader,
    AmazonOfficeSchoolSuppliesReader,
    AmazonPatioLawnGardenReader,
    AmazonPetSuppliesReader,
    AmazonPopReader,
    AmazonPublishersReader,
    AmazonPurchaseCirclesReader,
    AmazonRBReader,
    AmazonRapHipHopReader,
    AmazonRockReader,
    AmazonSoftwareReader,
    AmazonSportsCollectiblesReader,
    AmazonSportsOutdoorsReader,
    AmazonToolsHomeImprovementReader,
    AmazonToysGamesReader,
    AmazonVideoGamesReader,
    AmazonWineReader,
    Movielens100KReader,
    Movielens1MReader,
    Movielens10MReader,
    Movielens20MReader,
    MovielensHetrec2011Reader,
    BookCrossingReader,
    DatingReader,
    EpinionsReader,
    FilmTrustReader,
    FrappeReader,
    GowallaReader,
    Jester2Reader,
    MarketBiasAmazonReader,
    MarketBiasModClothReader,
    MovieTweetingsReader,
    NetflixPrizeReader,
    RecipesReader,
    WikilensReader,
    AnimeReader,
    CiaoDVDReader,
    GoogleLocalReviewsReader,
    LastFMReader,
]

DATASET_READER_NAME_LIST = [c.__name__ for c in DATASET_READER_LIST]

DATASET_DICT = {
    name: c for name, c in zip(DATASET_READER_NAME_LIST, DATASET_READER_LIST)
}

def dataset_handler(dataset_reader_name):
    """
    Returns:
        - dataset reader object
    """

    assert (
        dataset_reader_name in DATASET_READER_NAME_LIST
    ), f"dataset reader name not recognized: {dataset_reader_name}"

    return DATASET_DICT[dataset_reader_name]
