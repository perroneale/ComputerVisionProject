-- MySQL dump 10.13  Distrib 8.0.20, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: signadvisor
-- ------------------------------------------------------
-- Server version	8.0.20

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `sign`
--

DROP TABLE IF EXISTS `sign`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sign` (
  `idsign` int NOT NULL AUTO_INCREMENT,
  `photo` varchar(2853) NOT NULL,
  `review` float NOT NULL,
  `url` varchar(2853) NOT NULL,
  PRIMARY KEY (`idsign`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sign`
--

LOCK TABLES `sign` WRITE;
/*!40000 ALTER TABLE `sign` DISABLE KEYS */;
INSERT INTO `sign` VALUES (1,'la_prosciutteria.png',4.5,'https://www.tripadvisor.it/Restaurant_Review-g187801-d10044689-Reviews-La_Prosciutteria_Bologna-Bologna_Province_of_Bologna_Emilia_Romagna.html'),(2,'sonora.png',4,'https://www.tripadvisor.it/Restaurant_Review-g652002-d2264325-Reviews-Sonora-Galatina_Province_of_Lecce_Puglia.html'),(3,'rosticceria_moscara.jpg',4.5,'https://www.tripadvisor.it/Restaurant_Review-g652002-d2437976-Reviews-Rosticceria_Moscara-Galatina_Province_of_Lecce_Puglia.html'),(4,'pizza_mania.jpg',4,'https://www.tripadvisor.it/Restaurant_Review-g652002-d4055991-Reviews-Pizzamania-Galatina_Province_of_Lecce_Puglia.html'),(6,'portici_salentini.jpg',5,'https://www.tripadvisor.it/Restaurant_Review-g12664305-d13497907-Reviews-Portici_Salentini-Collemeto_Province_of_Lecce_Puglia.html'),(7,'pizzettari.jpg',4,'https://www.google.com/search?tbm=lcl&sxsrf=ALeKk03U4_pRbeUFH89AxoeNE3opINdLpA%3A1589820555422&ei=i7zCXvyjGaWRmwXSmKfQBg&q=i+pizzettari+aradeo&oq=i+pizzettari&gs_l=psy-ab.3.1.0l2j0i22i30k1j38.103928.106288.0.107799.12.12.0.0.0.0.262.1442.0j7j2.9.0....0...1c.1.64.psy-ab..3.9.1439...0i10k1j0i22i10i30k1.0.hjSeOfJ56C0#rlfi=hd:;si:4912895953175148308;mv:[[40.129648177319034,18.134424690621472],[40.12928822268097,18.133953909378523]]'),(8,'scapricciatiello.jpg',3.5,'https://www.tripadvisor.it/Restaurant_Review-g7791591-d1157986-Reviews-Scapricciatiello-Lido_Conchiglie_Gallipoli_Province_of_Lecce_Puglia.html');
/*!40000 ALTER TABLE `sign` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-21 10:00:06
