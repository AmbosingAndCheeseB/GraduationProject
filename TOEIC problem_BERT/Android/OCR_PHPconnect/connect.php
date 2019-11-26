<?php

  error_reporting(E_ALL);
  ini_set("display_errors", 1);

  $toeic = $_POST['toeic'];
  $toeic_split = explode('(', $toeic);

  $temp = str_replace("—","",$toeic_split[0]);
  $index = strpos($temp, "-");
  $temp = substr_replace($temp, " _ ", $index, 0);
  $question = str_replace("-","",$temp);

  $number1 = explode(')', $toeic_split[1]);
  $number2 = explode(')', $toeic_split[2]);
  $number3 = explode(')', $toeic_split[3]);
  $number4 = explode(')', $toeic_split[4]);

  $number_1 = $number1[1];
  $number_2 = $number2[1];
  $number_3 = $number3[1];
  $number_4 = $number4[1];

  echo $question;
  echo $number_1;
  echo $number_2;
  echo $number_3;
  echo $number_4;

?>
