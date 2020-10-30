# Lavaan Code for lav_Test.R
library(haven)
library(lavaan)
Test.datafile <- read_dta("./lav_Test.dta")
Test.model <- '
     # Dynamic Panel Data Model using ML for outcome variable conflict

     # Structural Equations
     conflict2 ~ 
          + c1*conflict1 
          + c2*ste_theta01

     conflict3 ~ 
          + c1*conflict2 
          + c2*ste_theta02

     conflict4 ~ 
          + c1*conflict3 
          + c2*ste_theta03

     conflict5 ~ 
          + c1*conflict4 
          + c2*ste_theta04

     conflict6 ~ 
          + c1*conflict5 
          + c2*ste_theta05

     conflict7 ~ 
          + c1*conflict6 
          + c2*ste_theta06

     conflict8 ~ 
          + c1*conflict7 
          + c2*ste_theta07

     conflict9 ~ 
          + c1*conflict8 
          + c2*ste_theta08

     conflict10 ~ 
          + c1*conflict9 
          + c2*ste_theta09

     conflict11 ~ 
          + c1*conflict10 
          + c2*ste_theta010

     conflict12 ~ 
          + c1*conflict11 
          + c2*ste_theta011

     conflict13 ~ 
          + c1*conflict12 
          + c2*ste_theta012

     conflict14 ~ 
          + c1*conflict13 
          + c2*ste_theta013

     conflict15 ~ 
          + c1*conflict14 
          + c2*ste_theta014

     conflict16 ~ 
          + c1*conflict15 
          + c2*ste_theta015

     conflict17 ~ 
          + c1*conflict16 
          + c2*ste_theta016

     conflict18 ~ 
          + c1*conflict17 
          + c2*ste_theta017

     conflict19 ~ 
          + c1*conflict18 
          + c2*ste_theta018

     conflict20 ~ 
          + c1*conflict19 
          + c2*ste_theta019

     conflict21 ~ 
          + c1*conflict20 
          + c2*ste_theta020

     conflict22 ~ 
          + c1*conflict21 
          + c2*ste_theta021

     conflict23 ~ 
          + c1*conflict22 
          + c2*ste_theta022

     conflict24 ~ 
          + c1*conflict23 
          + c2*ste_theta023

     conflict25 ~ 
          + c1*conflict24 
          + c2*ste_theta024

     conflict26 ~ 
          + c1*conflict25 
          + c2*ste_theta025

     conflict27 ~ 
          + c1*conflict26 
          + c2*ste_theta026

     conflict28 ~ 
          + c1*conflict27 
          + c2*ste_theta027

     conflict29 ~ 
          + c1*conflict28 
          + c2*ste_theta028

     conflict30 ~ 
          + c1*conflict29 
          + c2*ste_theta029

     conflict31 ~ 
          + c1*conflict30 
          + c2*ste_theta030

     conflict32 ~ 
          + c1*conflict31 
          + c2*ste_theta031

     conflict33 ~ 
          + c1*conflict32 
          + c2*ste_theta032

     conflict34 ~ 
          + c1*conflict33 
          + c2*ste_theta033

     conflict35 ~ 
          + c1*conflict34 
          + c2*ste_theta034

     conflict36 ~ 
          + c1*conflict35 
          + c2*ste_theta035

     conflict37 ~ 
          + c1*conflict36 
          + c2*ste_theta036

     conflict38 ~ 
          + c1*conflict37 
          + c2*ste_theta037

     conflict39 ~ 
          + c1*conflict38 
          + c2*ste_theta038


     # Alpha loadings equal 1 for all times
     Alpha =~ 
          + 1*conflict2 + 1*conflict3 + 1*conflict4 + 1*conflict5 + 1*conflict6 
          + 1*conflict7 + 1*conflict8 + 1*conflict9 + 1*conflict10 + 1*conflict11 
          + 1*conflict12 + 1*conflict13 + 1*conflict14 + 1*conflict15 + 1*conflict16 
          + 1*conflict17 + 1*conflict18 + 1*conflict19 + 1*conflict20 + 1*conflict21 
          + 1*conflict22 + 1*conflict23 + 1*conflict24 + 1*conflict25 + 1*conflict26 
          + 1*conflict27 + 1*conflict28 + 1*conflict29 + 1*conflict30 + 1*conflict31 
          + 1*conflict32 + 1*conflict33 + 1*conflict34 + 1*conflict35 + 1*conflict36 
          + 1*conflict37 + 1*conflict38 + 1*conflict39 

     # Fixed Effects Model - Alpha correlated with Time-Varying Exogenous Vars
     Alpha ~~ 
           + conflict1  + ste_theta01  + ste_theta02  + ste_theta03 
           + ste_theta04  + ste_theta05  + ste_theta06  + ste_theta07 
           + ste_theta08  + ste_theta09  + ste_theta010  + ste_theta011 
           + ste_theta012  + ste_theta013  + ste_theta014  + ste_theta015 
           + ste_theta016  + ste_theta017  + ste_theta018  + ste_theta019 
           + ste_theta020  + ste_theta021  + ste_theta022  + ste_theta023 
           + ste_theta024  + ste_theta025  + ste_theta026  + ste_theta027 
           + ste_theta028  + ste_theta029  + ste_theta030  + ste_theta031 
           + ste_theta032  + ste_theta033  + ste_theta034  + ste_theta035 
           + ste_theta036  + ste_theta037  + ste_theta038 

     # Correlations between Ys and predetermined variables
     ste_theta03 ~~
          + conflict2

     ste_theta04 ~~
          + conflict2 + conflict3

     ste_theta05 ~~
          + conflict2 + conflict3 + conflict4

     ste_theta06 ~~
          + conflict2 + conflict3 + conflict4 + conflict5

     ste_theta07 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6

     ste_theta08 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7

     ste_theta09 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8

     ste_theta010 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9

     ste_theta011 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10

     ste_theta012 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11

     ste_theta013 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12

     ste_theta014 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13

     ste_theta015 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14

     ste_theta016 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15

     ste_theta017 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16

     ste_theta018 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17

     ste_theta019 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18

     ste_theta020 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19

     ste_theta021 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20

     ste_theta022 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21

     ste_theta023 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22

     ste_theta024 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23

     ste_theta025 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24

     ste_theta026 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25

     ste_theta027 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26

     ste_theta028 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27

     ste_theta029 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28

     ste_theta030 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29

     ste_theta031 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30

     ste_theta032 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30 + conflict31

     ste_theta033 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30 + conflict31
          + conflict32

     ste_theta034 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30 + conflict31
          + conflict32 + conflict33

     ste_theta035 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30 + conflict31
          + conflict32 + conflict33 + conflict34

     ste_theta036 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30 + conflict31
          + conflict32 + conflict33 + conflict34 + conflict35

     ste_theta037 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30 + conflict31
          + conflict32 + conflict33 + conflict34 + conflict35 + conflict36

     ste_theta038 ~~
          + conflict2 + conflict3 + conflict4 + conflict5 + conflict6 + conflict7
          + conflict8 + conflict9 + conflict10 + conflict11 + conflict12 + conflict13
          + conflict14 + conflict15 + conflict16 + conflict17 + conflict18 + conflict19
          + conflict20 + conflict21 + conflict22 + conflict23 + conflict24 + conflict25
          + conflict26 + conflict27 + conflict28 + conflict29 + conflict30 + conflict31
          + conflict32 + conflict33 + conflict34 + conflict35 + conflict36 + conflict37

     # Constants constrained to be equal across time
     conflict2 ~ c3*1
     conflict3 ~ c3*1
     conflict4 ~ c3*1
     conflict5 ~ c3*1
     conflict6 ~ c3*1
     conflict7 ~ c3*1
     conflict8 ~ c3*1
     conflict9 ~ c3*1
     conflict10 ~ c3*1
     conflict11 ~ c3*1
     conflict12 ~ c3*1
     conflict13 ~ c3*1
     conflict14 ~ c3*1
     conflict15 ~ c3*1
     conflict16 ~ c3*1
     conflict17 ~ c3*1
     conflict18 ~ c3*1
     conflict19 ~ c3*1
     conflict20 ~ c3*1
     conflict21 ~ c3*1
     conflict22 ~ c3*1
     conflict23 ~ c3*1
     conflict24 ~ c3*1
     conflict25 ~ c3*1
     conflict26 ~ c3*1
     conflict27 ~ c3*1
     conflict28 ~ c3*1
     conflict29 ~ c3*1
     conflict30 ~ c3*1
     conflict31 ~ c3*1
     conflict32 ~ c3*1
     conflict33 ~ c3*1
     conflict34 ~ c3*1
     conflict35 ~ c3*1
     conflict36 ~ c3*1
     conflict37 ~ c3*1
     conflict38 ~ c3*1
     conflict39 ~ c3*1
     

     # Error variances free to vary across time
     conflict2 ~~ conflict2
     conflict3 ~~ conflict3
     conflict4 ~~ conflict4
     conflict5 ~~ conflict5
     conflict6 ~~ conflict6
     conflict7 ~~ conflict7
     conflict8 ~~ conflict8
     conflict9 ~~ conflict9
     conflict10 ~~ conflict10
     conflict11 ~~ conflict11
     conflict12 ~~ conflict12
     conflict13 ~~ conflict13
     conflict14 ~~ conflict14
     conflict15 ~~ conflict15
     conflict16 ~~ conflict16
     conflict17 ~~ conflict17
     conflict18 ~~ conflict18
     conflict19 ~~ conflict19
     conflict20 ~~ conflict20
     conflict21 ~~ conflict21
     conflict22 ~~ conflict22
     conflict23 ~~ conflict23
     conflict24 ~~ conflict24
     conflict25 ~~ conflict25
     conflict26 ~~ conflict26
     conflict27 ~~ conflict27
     conflict28 ~~ conflict28
     conflict29 ~~ conflict29
     conflict30 ~~ conflict30
     conflict31 ~~ conflict31
     conflict32 ~~ conflict32
     conflict33 ~~ conflict33
     conflict34 ~~ conflict34
     conflict35 ~~ conflict35
     conflict36 ~~ conflict36
     conflict37 ~~ conflict37
     conflict38 ~~ conflict38
     conflict39 ~~ conflict39
     

     # Exogenous variable covariances
     conflict1 ~~
        + ste_theta01 + ste_theta02 + ste_theta03 + ste_theta04
        + ste_theta05 + ste_theta06 + ste_theta07 + ste_theta08
        + ste_theta09 + ste_theta010 + ste_theta011 + ste_theta012
        + ste_theta013 + ste_theta014 + ste_theta015 + ste_theta016
        + ste_theta017 + ste_theta018 + ste_theta019 + ste_theta020
        + ste_theta021 + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta01 ~~
        + ste_theta02 + ste_theta03 + ste_theta04
        + ste_theta05 + ste_theta06 + ste_theta07
        + ste_theta08 + ste_theta09 + ste_theta010
        + ste_theta011 + ste_theta012 + ste_theta013
        + ste_theta014 + ste_theta015 + ste_theta016
        + ste_theta017 + ste_theta018 + ste_theta019
        + ste_theta020 + ste_theta021 + ste_theta022
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta02 ~~
        + ste_theta03 + ste_theta04 + ste_theta05
        + ste_theta06 + ste_theta07 + ste_theta08
        + ste_theta09 + ste_theta010 + ste_theta011
        + ste_theta012 + ste_theta013 + ste_theta014
        + ste_theta015 + ste_theta016 + ste_theta017
        + ste_theta018 + ste_theta019 + ste_theta020
        + ste_theta021 + ste_theta022 + ste_theta023
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta03 ~~
        + ste_theta04 + ste_theta05 + ste_theta06
        + ste_theta07 + ste_theta08 + ste_theta09
        + ste_theta010 + ste_theta011 + ste_theta012
        + ste_theta013 + ste_theta014 + ste_theta015
        + ste_theta016 + ste_theta017 + ste_theta018
        + ste_theta019 + ste_theta020 + ste_theta021
        + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta04 ~~
        + ste_theta05 + ste_theta06 + ste_theta07
        + ste_theta08 + ste_theta09 + ste_theta010
        + ste_theta011 + ste_theta012 + ste_theta013
        + ste_theta014 + ste_theta015 + ste_theta016
        + ste_theta017 + ste_theta018 + ste_theta019
        + ste_theta020 + ste_theta021 + ste_theta022
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta05 ~~
        + ste_theta06 + ste_theta07 + ste_theta08
        + ste_theta09 + ste_theta010 + ste_theta011
        + ste_theta012 + ste_theta013 + ste_theta014
        + ste_theta015 + ste_theta016 + ste_theta017
        + ste_theta018 + ste_theta019 + ste_theta020
        + ste_theta021 + ste_theta022 + ste_theta023
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta06 ~~
        + ste_theta07 + ste_theta08 + ste_theta09
        + ste_theta010 + ste_theta011 + ste_theta012
        + ste_theta013 + ste_theta014 + ste_theta015
        + ste_theta016 + ste_theta017 + ste_theta018
        + ste_theta019 + ste_theta020 + ste_theta021
        + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta07 ~~
        + ste_theta08 + ste_theta09 + ste_theta010
        + ste_theta011 + ste_theta012 + ste_theta013
        + ste_theta014 + ste_theta015 + ste_theta016
        + ste_theta017 + ste_theta018 + ste_theta019
        + ste_theta020 + ste_theta021 + ste_theta022
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta08 ~~
        + ste_theta09 + ste_theta010 + ste_theta011
        + ste_theta012 + ste_theta013 + ste_theta014
        + ste_theta015 + ste_theta016 + ste_theta017
        + ste_theta018 + ste_theta019 + ste_theta020
        + ste_theta021 + ste_theta022 + ste_theta023
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta09 ~~
        + ste_theta010 + ste_theta011 + ste_theta012
        + ste_theta013 + ste_theta014 + ste_theta015
        + ste_theta016 + ste_theta017 + ste_theta018
        + ste_theta019 + ste_theta020 + ste_theta021
        + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta010 ~~
        + ste_theta011 + ste_theta012 + ste_theta013
        + ste_theta014 + ste_theta015 + ste_theta016
        + ste_theta017 + ste_theta018 + ste_theta019
        + ste_theta020 + ste_theta021 + ste_theta022
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta011 ~~
        + ste_theta012 + ste_theta013 + ste_theta014
        + ste_theta015 + ste_theta016 + ste_theta017
        + ste_theta018 + ste_theta019 + ste_theta020
        + ste_theta021 + ste_theta022 + ste_theta023
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta012 ~~
        + ste_theta013 + ste_theta014 + ste_theta015
        + ste_theta016 + ste_theta017 + ste_theta018
        + ste_theta019 + ste_theta020 + ste_theta021
        + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta013 ~~
        + ste_theta014 + ste_theta015 + ste_theta016
        + ste_theta017 + ste_theta018 + ste_theta019
        + ste_theta020 + ste_theta021 + ste_theta022
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta014 ~~
        + ste_theta015 + ste_theta016 + ste_theta017
        + ste_theta018 + ste_theta019 + ste_theta020
        + ste_theta021 + ste_theta022 + ste_theta023
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta015 ~~
        + ste_theta016 + ste_theta017 + ste_theta018
        + ste_theta019 + ste_theta020 + ste_theta021
        + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta016 ~~
        + ste_theta017 + ste_theta018 + ste_theta019
        + ste_theta020 + ste_theta021 + ste_theta022
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta017 ~~
        + ste_theta018 + ste_theta019 + ste_theta020
        + ste_theta021 + ste_theta022 + ste_theta023
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta018 ~~
        + ste_theta019 + ste_theta020 + ste_theta021
        + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta019 ~~
        + ste_theta020 + ste_theta021 + ste_theta022
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta020 ~~
        + ste_theta021 + ste_theta022 + ste_theta023
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta021 ~~
        + ste_theta022 + ste_theta023 + ste_theta024
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta022 ~~
        + ste_theta023 + ste_theta024 + ste_theta025
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta023 ~~
        + ste_theta024 + ste_theta025 + ste_theta026
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta024 ~~
        + ste_theta025 + ste_theta026 + ste_theta027
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta025 ~~
        + ste_theta026 + ste_theta027 + ste_theta028
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta026 ~~
        + ste_theta027 + ste_theta028 + ste_theta029
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta027 ~~
        + ste_theta028 + ste_theta029 + ste_theta030
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta028 ~~
        + ste_theta029 + ste_theta030 + ste_theta031
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta029 ~~
        + ste_theta030 + ste_theta031 + ste_theta032
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta030 ~~
        + ste_theta031 + ste_theta032 + ste_theta033
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta031 ~~
        + ste_theta032 + ste_theta033 + ste_theta034
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta032 ~~
        + ste_theta033 + ste_theta034 + ste_theta035
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta033 ~~
        + ste_theta034 + ste_theta035 + ste_theta036
        + ste_theta037 + ste_theta038

     ste_theta034 ~~
        + ste_theta035 + ste_theta036 + ste_theta037
        + ste_theta038

     ste_theta035 ~~
        + ste_theta036 + ste_theta037 + ste_theta038

     ste_theta036 ~~
        + ste_theta037 + ste_theta038

     ste_theta037 ~~
        + ste_theta038


     # End of lavaan sem specification
     '

Test.results <- lavaan::sem(Test.model, 
   data = Test.datafile,
   missing = "fiml",
   estimator = "ML",
   se = "default",
   )
lavaan::summary(Test.results, fit.measures=FALSE)

