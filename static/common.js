document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.card');

    cards.forEach(card => {
        card.addEventListener('click', () => {
            // 모든 카드의 라디오 버튼을 비활성화
            cards.forEach(c => {
                c.querySelector('.radio-button').classList.remove('checked')
                c.classList.remove('selected')
            } );

            // 클릭된 카드의 라디오 버튼을 활성화
            card.querySelector('.radio-button').classList.add('checked');
            card.classList.add('selected');
        });
    });
});


document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.card');

    cards.forEach(card => {
        card.addEventListener('click', () => {
            // 모든 카드의 라디오 버튼 비활성화 및 선택 상태 해제
            cards.forEach(c => {
                c.querySelector('.radio-button').classList.remove('checked');
                c.classList.remove('selected');
            });

            // 클릭된 카드의 라디오 버튼 활성화 및 선택 상태 표시
            card.querySelector('.radio-button').classList.add('checked');
            card.classList.add('selected');
        });
    });
});
