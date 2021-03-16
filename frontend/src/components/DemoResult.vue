<template>
  <div class="demo_result">
    <div class="contrast_section">
      <div
        class="contrast_image_wrapper shadow"
        @mouseover="changeImage('in')"
        @mouseleave="changeImage('out')"
      >
        <div class="contrast_image card">
          <img class="image" :src="image.input" alt="input image" />
          <div class="image_info">
            <div class="image_name">original image</div>
            <div class="image_size">{{ sizeFormat(data.size.input) }}</div>
          </div>
        </div>
        <div class="contrast_image card" :class="{ hidden }">
          <img class="image" :src="contrastImage.src" :alt="contrastImage.name" />
          <div class="image_info">
            <div class="image_name">{{ contrastImage.name }} image</div>
            <div class="image_size">{{ sizeFormat(contrastImage.size) }}</div>
          </div>
        </div>
      </div>
      <div class="contrast_info card shadow">
        <div class="info_line clickable" @click="select('output')">
          <div class="selector" :class="{ selected: selectedImageName == 'output' }"></div>
          <div class="image_name">Our method</div>
          <div class="image_info">
            <div>{{ sizeFormat(data.size.fic) }} (compress data)</div>
            <div>compression ratio: {{ percentFormat(data.eval.fic_compression_ratio) }}</div>
            <div>PSNR: {{ data.eval.fic_psnr.toFixed(6) }}</div>
            <div>SSIM: {{ data.eval.fic_ssim.toFixed(6) }}</div>
          </div>
        </div>
        <div class="divider"></div>
        <div class="info_line clickable" @click="select('jpeg')">
          <div class="selector" :class="{ selected: selectedImageName == 'jpeg' }"></div>
          <div class="image_name">JPEG</div>
          <div class="image_info">
            <div>{{ sizeFormat(data.size.jpeg) }}</div>
            <div>compression ratio: {{ percentFormat(data.eval.jpeg_compression_ratio) }}</div>
            <div>PSNR: {{ data.eval.jpeg_psnr.toFixed(6) }}</div>
            <div>SSIM: {{ data.eval.jpeg_ssim.toFixed(6) }}</div>
          </div>
        </div>
      </div>
    </div>
    <div class="card_wrapper" v-viewer="{ filter: excludeIcon }">
      <!-- TODO 提供下载/查看原图功能 -->
      <ImageCard class="card clickable" :src="image.input" name="input"></ImageCard>
      <ImageCard class="card clickable" :src="image.feat" name="feature"></ImageCard>
      <ImageCard class="card clickable" :src="image.output" name="output"></ImageCard>
      <ImageCard class="card clickable" :src="image.jpeg" name="jpeg"></ImageCard>
      <ImageCard class="card clickable" :src="image.resi" name="residual"></ImageCard>
      <ImageCard class="card clickable" :src="image.recon" name="reconstruction"></ImageCard>
      <ImageCard
        class="card clickable"
        :src="image.resi_norm"
        name="normalize residual"
      ></ImageCard>
      <ImageCard
        class="card clickable"
        :src="image.recon_norm"
        name="normalize reconstruction"
      ></ImageCard>
      <ImageCard
        class="card clickable"
        name="compress data"
        src="/assets/archive.png"
        is-icon
        @click.native="downloadFic"
      ></ImageCard>
    </div>
  </div>
</template>

<script>
import ImageCard from '@/components/ImageCard.vue';

export default {
  props: {
    data: {
      type: Object,
      required: true,
    },
  },
  data() {
    return {
      selectedImageName: 'output',
      hidden: false,
    };
  },
  computed: {
    image() {
      return this.data.image;
    },
    contrastImage() {
      return {
        name: this.selectedImageName,
        src: this.image[this.selectedImageName],
        size: this.data.size[this.selectedImageName],
      };
    },
  },
  mounted() {
    const cardWrapper = this.$el.querySelector('.card_wrapper');
    cardWrapper.addEventListener('ready', () => {
      this.viewer = cardWrapper.$viewer;
    });
  },
  methods: {
    changeImage(type) {
      this.hidden = type === 'in';
    },
    select(name) {
      this.selectedImageName = name;
    },
    excludeIcon(e) {
      return ![...e.classList].includes('icon');
    },
    downloadFic() {
      window.open(this.data.data);
    },
  },
  components: {
    ImageCard,
  },
};
</script>

<style lang="scss" scoped>
.demo_result {
  $margin: 20px;
  margin: 0 auto;

  .contrast_section {
    display: grid;
    grid-template-columns: 38.2% 61.8%;
    margin: $margin 0;

    .contrast_image_wrapper {
      max-width: 100%;
      max-height: 100%;
      margin-right: $margin;

      position: relative;

      .contrast_image {
        transition: opacity var(--duration);
        width: 100%;

        .image {
          width: 100%;
          height: 100%;
        }

        &.hidden {
          opacity: 0;
        }

        &:nth-of-type(2) {
          position: absolute;
          top: 0;
          left: 0;
        }
      }

      .image_info {
        display: flex;
        justify-content: space-between;
        margin: 12px;
      }
    }

    .contrast_info {
      margin-left: $margin;
      padding: 50px 60px;

      display: flex;
      flex-direction: column;
      justify-content: space-around;

      .info_line {
        display: flex;
      }

      .selector {
        width: 40px;
        height: 26px;
        flex: none;
        position: relative;

        &:before {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);

          height: 8px;
          width: 8px;
          border: 1px solid var(--border);
          box-sizing: border-box;
          border-radius: 50%;
          transition: all var(--duration);
        }

        &.selected:before {
          height: 14px;
          width: 14px;
          border: none;
          background: var(--primary);
        }
      }

      .image_name {
        color: var(--text);
        font-size: 22px;
        line-height: 1.2;
        position: relative;
        flex: 1.2;
      }

      .image_info {
        font-size: 16px;
        line-height: 1.5;
        color: var(--text2);
        flex: 2;
      }

      .divider {
        border-top: var(--border) solid 1px;
      }
    }
  }

  .card_wrapper {
    padding: 40px 0;
    display: grid;
    grid-template-columns: repeat(4, 22.5%);
    gap: 3.33%;
  }
}
</style>
