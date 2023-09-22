$( document ).ready(function() {
    
    toastr.options = {
        "debug": false,
        "positionClass": "toast-bottom-full-width",
        "onclick": null,
        "fadeIn": 300,
        "fadeOut": 1000,
        "timeOut": 2000,
        "extendedTimeOut": 1000
      }

    var board = [11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44];
    var suits = [1,2];
    var selectedCards = [];
    var playAgainst = 0;
    var aiSelectedCards = [];
    
    $("#selAgent").on('change', function(e) {
        playAgainst = Number(this.value);
    })

    $("#selUserSuit").on('change', function(e) {
        suits[0] = Number(this.value);
    })

    $("#selAgentSuit").on('change', function(e) {
        suits[1] = Number(this.value);
    })

    $("#bNewGame").on('click',function (){
        board = shuffleArray(board);
        selectedCards = [];
        aiSelectedCards = [];
        resetStyleForAllImages();
        displayBoard();
    });

    $(".img-container").on('click', function(e) {
        //console.log(e);
        resetStyleForAllImages();
        //console.log(selectedCards);
        if(selectedCards.length>1) {
            selectedCards.shift();
        }
        var idStr = $(e.target).attr('id')
        if(idStr.charAt(idStr.length-1) == 'i')
            var target = Number(idStr.substring(1, idStr.length-1))
        else 
        var target = Number(idStr.substring(1, idStr.length))
        selectedCards.push(target);
        //console.log(selectedCards);
        //console.log(idStr,target);
        highlightSelectedImages();
        
    });

    function shuffleArray(array) { 
        //return array.sort( ()=>Math.random()-0.5 );
        let currentIndex = array.length,  randomIndex;

        while (currentIndex != 0) {
          randomIndex = Math.floor(Math.random() * currentIndex);
          currentIndex--;
          [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
        }
      
        return array;
    } 

    function displayBoard() {
        var i = 0;
        var size = board.length;
        while(i<size) {
            $("#p"+i+"i").attr("src",getImage(board[i]));
            i++;
        }
    }

    function getImage(i) {
        return "img/"+i+".png";
    }

    function resetStyleForAllImages() {
        var i = 0;
        var size = board.length;
        while(i<size) {
            $("#p"+i).attr("style","");
            i++;
        }
    }

    function highlightSelectedImages() {
        var i = 0, len = selectedCards.length;
        while (i < len) {
            $("#p"+selectedCards[i]).attr("style","border: 5px solid #ef476f");
            //06d6a0, 118ab2,  ffd166
            i++
        }

        i=0; len = aiSelectedCards.length;
        while (i < len) {
            $("#p"+aiSelectedCards[i]).attr("style","border: 5px solid #06d6a0");
            //06d6a0, 118ab2,  ffd166
            i++
        }

    }
    //toastr.success('Success messages',);
    $("#btPlayTurn").on('click', function() {
        postData = { "board": board,
                 "suits": suits,
                 "selectedCards": selectedCards,
                 "playAgainst": playAgainst
                }
        $.ajax({
            url: '/',
            type: 'POST',
            data: JSON.stringify(postData),
            success: function(response) {
                console.log(response);
                if(response.processPlay) {
                    aiSelectedCards = []
                    if(response.board != null) board = response.board
                    if(response.aiSelectedCards != null)  aiSelectedCards = response.aiSelectedCards
                    if(response.msg_e) toastr.error(response.msg_e)
                    if(response.msg_i) toastr.info(response.msg_i)
                    if(response.msg_s) toastr.success(response.msg_s)
                    resetStyleForAllImages();
                    displayBoard();
                    highlightSelectedImages();
                }
            }
         });
    });




});