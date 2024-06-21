$(document).ready(function() {
    const $cards = $('.card');
    let check_data_arr = {}; // 선택한 데이터의 값들이 저장될 객체

    const links = $('.main_menu ul li');
    const contentDiv = $('#main');

    // Function to load content from HTML file
    function loadContent(url) {
        $.ajax({
            url: url,
            method: 'GET',
            dataType: 'html',
            success: function(data) {
                contentDiv.html(data);
            },
            error: function() {
                contentDiv.html('<p>콘텐츠 로드 오류.</p>');
            }
        });
    }

    // Load default content (home.html)
    loadContent('learning');

    // Add event listeners to links
    links.on('click', function(event) {
        event.preventDefault();
        const contentUrl = $(this ).find('a').data('content');
        loadContent(contentUrl);

        // 메뉴 클릭시 스타일 효과 

        $('a').css("color", "");
        $(".main_menu ul li").css("background-color" , "")
        $('.main_menu ul li > a > img').css("filter","");

        $(this).css("background-color" , "#2B2D53")
        $(this).find('a').css("color", "#1ADEC2");
        $(this).find('a > img').css("filter", "invert(85%) sepia(85%) saturate(5323%) hue-rotate(90deg) brightness(96%) contrast(80%)");

        console.log(this)
    });

    // Add event listeners to cards
    $(document).on('click', '.card', function() {
        // 클릭된 카드의 라디오 버튼을 활성화
        $(this).find('.radio-button').addClass('checked');
        $(this).addClass('selected');
    
        // 나머지 카드의 라디오 버튼을 비활성화
        $('.card').not(this).find('.radio-button').removeClass('checked');
        $('.card').not(this).removeClass('selected');
    
        // 선택된 모델을 check_data_arr에 저장
        check_data_arr.model_name = $.trim($(this).text());
        console.log("Selected model:", check_data_arr.model_name);
    });
    
    $(document).on('click', '#learning_start_btn', function() {
        const batch_size = $("#batch_size").val()
        const lr = $("#lr").val()
        const stop_number = $("#stop_number").val()
        const file_name = $("#file_name").val()
        
        const epochs = $("#epochs").val()
        const num_workers = $("#num_workers").val()
        
        // data input event
        check_data_arr.batch_size = $.trim(batch_size);
        check_data_arr.lr = $.trim(lr);
        check_data_arr.stop_number = $.trim(stop_number);
        check_data_arr.file_name = $.trim(file_name);
        check_data_arr.epochs = $.trim(epochs);
        check_data_arr.num_workers = $.trim(num_workers);

        // 로딩 바 나타나기 
        $("#loader_box").css("display", "block")
        $("#spinner").css("display", "block")

        console.log(JSON.stringify(check_data_arr, null, 2 ))
        $.ajax({
            url: '/train/parameters',
            method: 'post',
            data : JSON.stringify(check_data_arr),
            dataType: 'json',
            contentType: 'application/json; charset=utf-8', 
            success: function (data, status, xhr) {
                console.log("data : : " + data);
                console.log("data : : " + JSON.stringify(data));

                // 성공 시 로딩 바 숨기기
                $("#loader_box").css("display", "none");
                $("#spinner").css("display", "none");
                alert("모델이 성공적으로 만들어졌어요!\n 모델 테스트하기 탭으로 이동해 모델의 성능을 평가해조세요!")
                setInterval(function() {
                    $('#tab02').toggleClass('blink');
                }, 5000);
            },
            error: function (data, status, err) {
                console.log(data)
            }
        });

    });



    $(document).on('click', '#test_start_btn', function() {
        const batch_size = $("#batch_size").val()
        const num_workers = $("#num_workers").val()
        
        // data input event
        check_data_arr.batch_size = $.trim(batch_size);
        check_data_arr.num_workers = $.trim(num_workers);

        // 로딩 바 나타나기 
        $("#loader_box").css("display", "block")
        $("#spinner").css("display", "block")

        console.log(JSON.stringify(check_data_arr, null, 2 ))
        $.ajax({
            url: '/test/parameters',
            method: 'post',
            data : JSON.stringify(check_data_arr),
            dataType: 'json',
            contentType: 'application/json; charset=utf-8', 
            success: function (data, status, xhr) {
                console.log("data : : " + data.f1_last_data);
                console.log("data : : " + JSON.stringify(data.f1_last_data));

                // 성공 시 로딩 바 숨기기
                $("#loader_box").css("display", "none");
                $("#spinner").css("display", "none");
                alert("모델이 성공적으로 만들어졌어요!\n 모델 테스트하기 탭으로 이동해 모델의 성능을 평가해조세요!")
                setInterval(function() {
                    $('#tab02').toggleClass('blink');
                }, 5000);



                const newElement = `
                <div class="setting_box">
                    모델의 파라미터를 지정해주세요.
                    <br> 
                    <div>
                        정확도 : `+data.f1_last_data+`
                    </div>
                </div>
                `;
                // 부모 요소에 새로운 요소 추가
                $('.setting_bottom_box').append(newElement);
            },
            error: function (data, status, err) {
                $("#loader_box").css("display", "none");
                $("#spinner").css("display", "none");
                console.log(data)
            }
        });

    });



    
    // 
    // lr
    // stop_number
    // file_name

    // 학습 시작 버튼 클릭시 발생하는 이벤트 
    // 클릭된 요소의 이름을 ajax데이터로 돌려야함.

    // 모델 학습 시키기 박스 클릭 이벤트
    $(document).on('click', '.step_boxs', function() {
        console.log($(this).find(".title_box").text())
        // $(this).find(".title_box").text("새로운 제목");
        if($(this).find(".title_box").text() == "모델 학습시키기"){
            $(".title > h2").text("모델 학습시키기")
        }else if($(this).find(".title_box").text() == "모델 테스트하기"){
            $(".title > h2").text("모델 테스트하기")

        }else{

            $(".title > h2").text("직접 데이터 넣어보기")
        }
        console.log($(this))
    })

    $('.tabcontent > div').hide();
    // $('#tab02').css("display","none");
    $(document).on('click', '.tabnav a', function() {
        check_data_arr = {}
        $('.tabcontent > div').hide().filter(this.hash).fadeIn();
        $('.tabnav a').removeClass('active');
        $(this).addClass('active');
        return false;
    }).filter(':eq(0)').click();

    // 파일 목록 가져오기 
    function loadFileList() {
        $.ajax({
            url: '/fileList', // 파일 목록을 제공하는 API 엔드포인트 URL
            method: 'get',
            dataType: 'json',
            success: function (data, status, xhr) {
                let fileContentArray = [];
                $('#file_list').empty(); // 기존 파일 목록 비우기
                $('#file_list2').empty(); // 기존 파일 목록 비우기

                for (let i = 0; i < data.length; i++) {
                    // 각 파일에 대한 HTML 요소 생성
                    const newElement = `
                        <div class="card" title="${data[i]}">
                            <div class="radio-button"></div>
                            <img src="../static/file2_icon.png" alt="Card Image" class="card-image file_image">
                            <div>
                                <h2 class="file-title">${data[i]}</h2>
                            </div>
                        </div>
                    `;
                    fileContentArray.push(newElement);
                }

                // 배열을 문자열로 합치기
                let combinedElements = fileContentArray.join('');

                // 파일 목록을 화면에 추가
                $('#file_list').html(combinedElements);
                $('#file_list2').html(combinedElements);
            },
            error: function (data, status, err) {
                console.log(data);
            }
        });
    }

    // 페이지 초기 로딩 시 파일 목록 불러오기
    loadFileList();

    // 페이지 리로드 시에도 파일 목록을 갱신하기 위해 동일한 함수 호출
    $(document).on('click', '.main_menu > ul > li', function() {
        loadFileList();
    });


    // 파일 업로드
    $(document).on('change', '#uploadFile', function() {
        const files = event.target.files;
        $('#imgArea').empty(); // 이전에 추가된 이미지를 지움
        $("#no_file_bg").hide()
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (file.type.startsWith('image/')) {
                const image = new Image();
                const ImageTempUrl = window.URL.createObjectURL(file);
                image.src = ImageTempUrl;
                image.classList.add('preview-image');
                $('#imgArea').append(image);

                // 메모리 누수를 방지하기 위해 이미지 로드 후 URL을 해제
                image.onload = function() {
                    window.URL.revokeObjectURL(ImageTempUrl);
                };
            }
        }
    });

    $(document).on('click', '#ajaxCall', async function() {
        // const form = document.forms.frm;
        let $form = document.frm;
        const fData = new FormData();
        alert($form[0])
        alert($form[0].files[0])
        fData.append("file_lo", $form[0].files[0]);
        // fData.append("model_pt_file_name", JSON.stringify(check_data_arr));
        fData.append("model_pt_file_name", new Blob([JSON.stringify(check_data_arr)], {type: "application/json"}));

        console.log(" $form[0]==>",  $form[0].files[0])
        console.log(" JSON.stringify(check_data_arr)==>",  JSON.stringify(check_data_arr, null, 2))

      
                
        $.ajax({
            url: '/upload',
            method: 'POST',
            data: fData,
            processData: false, // 파일 데이터를 문자열로 변환하지 않도록 설정
            contentType: false, // 기본 설정된 content-type 헤더를 제거
            success: function (data, status, xhr) {
                console.log("file name " ,JSON.stringify(data.cat, null, 2))
                console.log("file name " ,JSON.stringify(data.dog, null, 2))
                // 성공 시 로딩 바 숨기기x

                $("#ma").text("고양이:" + JSON.stringify(data.cat, null, 2).slice(0, 5)+"%")
                $("#eum").text("강아지 :" + JSON.stringify(data.dog, null, 2).slice(0, 5)+"%")

                
            },
            error: function (data, status, err) {
                console.log(data)
            }
        });
    })  



    // async function _post(path, bodyData = {}) {
    //     const response = await fetch(path, {
    //         method: 'POST',
    //         body: bodyData
    //     });
    //     return response.json();
    // }
    
    // $(document).on('click', '#ajaxCall', async function() {
    //     const form = document.forms.frm;
    //     const fileInput = form.querySelector('input[type="file"]');
    //     const fData = new FormData();
    //     fData.append("file_lo", fileInput.files[0]);
    
    //     const ma = document.querySelector("#ma");
    //     const eum = document.querySelector("#eum");
    //     const toial_max = document.querySelector("#toial_max");
    //     console.log(fData)
    //     const rvalue = await _post('/upload', fData);
    
    //     ma.innerHTML = `마동석 : ${rvalue.ma_data}`;
    //     eum.innerHTML = `음문석 : ${rvalue.eum_data}`;
    
    //     let person;
    //     switch (rvalue.max_data) {
    //         case "0":
    //             person = "음문석";
    //             break;
    //         case "1":
    //             person = "마동석";
    //             break;
    //         default:
    //             person = "알 수 없는 사람";
    //     }
    //     toial_max.innerHTML = `이미지의 사람은 ${person}입니다.`;
    
    //     console.log("결과 값=====================");
    //     console.log(rvalue.max_data === "0");
    //     console.log(rvalue);
    // });
    
});
