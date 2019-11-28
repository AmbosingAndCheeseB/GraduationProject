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

  $sentence = $question."/".$number1[1]."/".$number2[1]."/".$number3[1]."/".$number4[1];

  $py = 'C:\파이썬 실행파일 절대경로\python.exe';
  $pysc = 'C:\파이썬 스크립트 파일 절대경로\toeic_bert.py --problem $sentence';
  $cmd = "$py $pysc";

  exec("$cmd $sentence", $out, $status);
  echo $out[0];

?>
