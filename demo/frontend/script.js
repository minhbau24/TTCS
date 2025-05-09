// Biến toàn cục
let currentPage = 0;
let currentCategory = 'sach';
let breadcrumbTrail = [];
const limit = 12;
const selectedProducts = [];

async function fetchCategories(category) {
    let categoryTree = [];
    const storedCategories = localStorage.getItem(`categories_${category}`);
    if (storedCategories) {
        try {
            categoryTree = JSON.parse(storedCategories);
            if (!Array.isArray(categoryTree)) {
                console.error('Stored categories is not an array:', categoryTree);
                categoryTree = [];
            }
        } catch (error) {
            console.error('Error parsing stored categories:', error);
            localStorage.removeItem(`categories_${category}`);
        }
    }

    if (!categoryTree.length) {
        try {
            const response = await fetch(`http://127.0.0.1:8000/categories?category=${encodeURIComponent(category)}`);
            if (!response.ok) throw new Error('Network response was not ok');
            categoryTree = await response.json();
            if (!Array.isArray(categoryTree)) {
                console.error('API response is not an array:', categoryTree);
                categoryTree = [];
            }
            localStorage.setItem(`categories_${category}`, JSON.stringify(categoryTree));
        } catch (error) {
            console.error('Error fetching categories:', error);
            return [];
        }
    }

    return categoryTree;
}

function formatCategoryName(name) {
    if (typeof name !== 'string') return 'Danh mục không xác định';
    return name.replace(/-/g, ' ')
               .toLowerCase()
               .split(' ')
               .map(word => word.charAt(0).toUpperCase() + word.slice(1))
               .join(' ');
}

function renderBreadcrumb() {
    const breadcrumbContainer = document.getElementById('breadcrumb');
    breadcrumbContainer.innerHTML = '';

    breadcrumbTrail.forEach((cat, index) => {
        const crumb = document.createElement('span');
        crumb.innerHTML = `
            <a href="#" class="breadcrumb-link" data-index="${index}">
                ${formatCategoryName(cat)}
            </a>
            ${index < breadcrumbTrail.length - 1 ? ' > ' : ''}
        `;
        breadcrumbContainer.appendChild(crumb);
    });

    document.querySelectorAll('.breadcrumb-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const index = parseInt(link.getAttribute('data-index'));
            const selectedCategory = breadcrumbTrail[index];
            const newPath = breadcrumbTrail.slice(0, index);
            handleCategoryClick(selectedCategory, newPath);
        });
    });
}

function renderSelectedProducts() {
    const selectedList = document.getElementById('selected-products');
    selectedList.innerHTML = '';

    selectedProducts.forEach(product => {
        const item = document.createElement('div');
        item.className = 'd-flex align-items-center mb-2';

        const name = document.createElement('span');
        name.className = 'me-2';
        name.textContent = product.name;

        const removeBtn = document.createElement('button');
        removeBtn.className = 'btn btn-sm btn-outline-danger';
        removeBtn.innerHTML = '&times;';
        removeBtn.title = 'Hủy chọn sản phẩm';

        removeBtn.addEventListener('click', () => {
            const index = selectedProducts.findIndex(p => p.id === product.id);
            if (index !== -1) {
                selectedProducts.splice(index, 1);
                renderSelectedProducts();

                const btns = document.querySelectorAll(`.product-select-btn[data-id="${product.id}"]`);
                btns.forEach(btn => {
                    btn.textContent = 'Chọn sản phẩm';
                    btn.className = 'btn btn-primary mt-auto product-select-btn';
                });
            }
        });

        item.appendChild(name);
        item.appendChild(removeBtn);
        selectedList.appendChild(item);
    });
}

async function loadProducts(page, limit, category) {
    const productsDiv = document.getElementById('products');
    try {
        productsDiv.innerHTML = Array(limit).fill().map(() => `
            <div class="col-md-3 mb-4">
                <div class="card h-100">
                    <div class="card-img-top bg-light" style="height: 200px;"></div>
                    <div class="card-body">
                        <div class="card-title bg-light" style="height: 20px; width: 80%;"></div>
                        <div class="card-text bg-light" style="height: 20px; width: 50%;"></div>
                    </div>
                </div>
            </div>
        `).join('');

        const response = await fetch(`http://localhost:8000/products?limit=${limit}&page=${page}&category=${encodeURIComponent(category)}`);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const { total, data } = await response.json();
        if (!Array.isArray(data)) throw new Error('Dữ liệu không hợp lệ: "data" không phải là mảng');

        productsDiv.innerHTML = '';
        if (data.length === 0) {
            productsDiv.innerHTML = '<p>Không có sản phẩm nào trong danh mục này.</p>';
        } else {
            data.forEach(product => {
                const productElement = document.createElement('div');
                productElement.className = 'col-md-3 mb-4';

                const card = document.createElement('div');
                card.className = 'card h-100';

                const img = document.createElement('img');
                img.src = product.image_url;
                img.className = 'card-img-top';
                img.alt = product.name;
                img.width = 200;
                img.height = 200;
                img.loading = 'lazy';

                const cardBody = document.createElement('div');
                cardBody.className = 'card-body d-flex flex-column';

                const title = document.createElement('h5');
                title.className = 'card-title';
                title.textContent = product.name;

                const price = document.createElement('p');
                price.className = 'card-text';
                price.textContent = `Giá: ${product.price} VND`;

                const isSelected = selectedProducts.some(p => p.id === product.id);
                const button = document.createElement('button');
                button.className = isSelected ? 'btn btn-danger mt-auto product-select-btn' : 'btn btn-primary mt-auto product-select-btn';
                button.textContent = isSelected ? 'Hủy chọn' : 'Chọn sản phẩm';
                button.dataset.id = product.id;

                button.addEventListener('click', () => {
                    const index = selectedProducts.findIndex(p => p.id === product.id);
                    const isSelected = index !== -1;

                    if (isSelected) {
                        selectedProducts.splice(index, 1);
                        button.textContent = 'Chọn sản phẩm';
                        button.className = 'btn btn-primary mt-auto product-select-btn';
                    } else {
                        selectedProducts.push(product);
                        button.textContent = 'Hủy chọn';
                        button.className = 'btn btn-danger mt-auto product-select-btn';
                    }

                    renderSelectedProducts();
                });

                cardBody.appendChild(title);
                cardBody.appendChild(price);
                cardBody.appendChild(button);
                card.appendChild(img);
                card.appendChild(cardBody);
                productElement.appendChild(card);
                productsDiv.appendChild(productElement);
            });
        }

        const paginationDiv = document.createElement('div');
        paginationDiv.className = 'd-flex justify-content-between mt-4';

        const prevButton = document.createElement('button');
        prevButton.className = 'btn btn-primary';
        prevButton.textContent = 'Previous';
        prevButton.disabled = page === 0;
        prevButton.addEventListener('click', () => {
            if (currentPage > 0) {
                currentPage--;
                loadProducts(currentPage, limit, currentCategory);
            }
        });
        paginationDiv.appendChild(prevButton);

        const nextButton = document.createElement('button');
        nextButton.className = 'btn btn-primary';
        nextButton.textContent = 'Next';
        nextButton.disabled = (page + 1) * limit >= total;
        nextButton.addEventListener('click', () => {
            currentPage++;
            loadProducts(currentPage, limit, currentCategory);
        });
        paginationDiv.appendChild(nextButton);

        productsDiv.appendChild(paginationDiv);
    } catch (error) {
        console.error('Error loading products:', error);
        productsDiv.innerHTML = `Error loading products: ${error.message}`;
    }
}

async function handleCategoryClick(category, path = []) {
    currentCategory = category;
    currentPage = 0;
    breadcrumbTrail = path.concat([category]);
    renderBreadcrumb();
    const categories = await fetchCategories(category);
    displayCategories(categories);
    loadProducts(currentPage, limit, currentCategory);
}

function displayCategories(categories) {
    const categoryList = document.getElementById('category-list');
    categoryList.innerHTML = '';

    let level1Categories = [];
    if (categories && Array.isArray(categories.children)) {
        level1Categories = categories.children;
    } else if (Array.isArray(categories)) {
        level1Categories = categories;
    }

    if (!Array.isArray(level1Categories) || level1Categories.length === 0) {
        categoryList.innerHTML = '<p>Không có danh mục nào để hiển thị.</p>';
        return;
    }

    level1Categories.forEach((category, index) => {
        if (!category || typeof category.name !== 'string') {
            console.warn('Invalid category object:', category);
            return;
        }

        const collapseId = `collapse-${index}`;
        const formattedName = formatCategoryName(category.name);

        const card = document.createElement('div');
        card.className = 'accordion-item mb-2';
        card.innerHTML = `
            <h2 class="accordion-header" id="heading-${index}">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}">
                    ${formattedName}
                </button>
            </h2>
            <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="heading-${index}" data-bs-parent="#category-list">
                <div class="accordion-body">
                    <ul class="list-group list-group-flush">
                        ${(Array.isArray(category.children) ? category.children : []).map((child, childIndex) => {
                            if (typeof child !== 'string') return '';
                            const formattedChildName = formatCategoryName(child);
                            return `
                                <li class="list-group-item">
                                    <a href="#" class="text-decoration-none category-link" data-category="${child}">
                                        ${formattedChildName}
                                    </a>
                                </li>
                            `;
                        }).join('')}
                    </ul>
                </div>
            </div>
        `;
        categoryList.appendChild(card);
    });

    document.querySelectorAll('.category-link').forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const category = link.getAttribute('data-category');
            handleCategoryClick(category, [...breadcrumbTrail]);
        });
    });
}

document.addEventListener('DOMContentLoaded', async function() {
    document.getElementById("submit-button").addEventListener("click", () => {
        // Lưu dữ liệu vào localStorage để trang ketqua.html dùng
        localStorage.setItem("selectedProducts", JSON.stringify(selectedProducts));

        // Chuyển sang trang kết quả (nơi sẽ gửi request và hiển thị)
        window.location.href = "result.html";
    });
    try {
        const categories = await fetchCategories('sach');
        breadcrumbTrail = ['sach'];
        renderBreadcrumb();
        displayCategories(categories);
        loadProducts(currentPage, limit, currentCategory);
    } catch (error) {
        console.error('Error in DOMContentLoaded:', error);
        document.getElementById('category-list').innerHTML = `<p>Lỗi khi tải danh mục: ${error.message}</p>`;
    }
});