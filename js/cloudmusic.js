function cloudmusic(){
arrMusicID = [33469292,32648543,28287132,26511034,435305771,27588029,41462985,28656150,4010799];  //musicID array
musicID = Math.floor(Math.random()*(arrMusicID.length)) //get a ran num as index
$('body').css('height',document.documentElement.clientHeight -5);

if (!Number.isInteger(arrMusicID[musicID])) return; // load failed, bye~

var iframe = document.createElement('iframe');
iframe.id="bgm";
iframe.style = "position: absolute; bottom: 0; left: 30px; border: 0px;";
iframe.src = '//music.163.com/outchain/player?type=2&id=' +arrMusicID[musicID]+ '&auto=1&height=32';
console.log(iframe.src)
iframe.frameborder="no";
iframe.marginwidth="0";
iframe.marginheight="0";
iframe.width=250;
iframe.height=52;
document.body.appendChild(iframe);
}
