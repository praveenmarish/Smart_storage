function myFunction() {
    var x = document.getElementById("image");
    if (x.src==location.origin+"/movement"){
        x.src=location.origin +"/video_feed";
    }
    else if(x.src==location.origin+"/video_feed"){
        x.src=location.origin +"/classify";
    }
    else{
        x.src=location.origin+"/movement"
    }
  }